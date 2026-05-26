import torch
import numpy as np
from astropy.time import Time
from typing import Union, Optional, Tuple, List, Sequence, TypeVar, Generic
from pathlib import Path
from abc import ABC, abstractmethod

Interval = Tuple[Union[float, int], Union[float, int]]
ArrayLike = Union[np.ndarray, torch.Tensor, Sequence, float, int, np.number]
IEnergyList = List[Union[float, int, Interval]]
MEnergyList = List[Interval]

T_IntVarList = TypeVar('T_IntVarList')
T_NormDensityType = TypeVar('T_NormDensityType')

from .NFBase import NFBase

class NFNormalizationMapBase(ABC, Generic[T_IntVarList, T_NormDensityType]):
    
    # Need to be defined by implementation
    _intvar_allow_float: bool
    _intvar_name: str
    
    def __init__(self,
                 nfdensity: Optional[T_NormDensityType] = None,
                 intvar: Optional[T_IntVarList] = None,
                 menergy_keV: Optional[MEnergyList] = None,
                 scatt_angle_rad: Optional[MEnergyList] = None,
                 ):
        
        self._nfdensity = nfdensity
        self._intvar_input = intvar
        self._menergy_keV_input = menergy_keV
        self._scatt_angle_rad_input = scatt_angle_rad 
        
        self._coordinates = None
        self._maps = None
        self._intvar = None
        self._menergy_keV = None
        self._scatt_angle_rad = None
        self._interpolator = None
        
        self._range_intvar: Optional[Tuple[float, float]] = None
        self._intvar_resolution: Optional[float] = None
        self._atol: float = 0.0
        self._menergy_keV_resolution: Optional[float] = None
        
        if nfdensity is not None:
            self._set_norm_dimensions(nfdensity.norm_dimensions)
        else:
            self._norm_dimensions = None
    
    @classmethod
    def from_cache(cls, filename: Union[str, Path]) -> T_NormDensityType:
        instance = cls()
        instance.cache_from_file(filename)
        return instance
    
    @property
    def atol(self):
        return self._atol
    @atol.setter
    def atol(self, val: float): self._set_integration_parameters(atol=val)
    
    @property
    def norm_dimensions(self):
        return self._norm_dimensions
    
    def _set_integration_parameters(self,
                                   intvar_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[Tuple[float, float]] = None,
                                   atol: Optional[float] = -1.0):
        
        new_intvar_resolution = intvar_resolution if intvar_resolution != -1.0 else self._intvar_resolution
        new_range_intvar = range_intvar or self._range_intvar
        new_atol = atol if atol != -1.0 else self._atol
        
        if not isinstance(new_intvar_resolution, (float, int, np.number)) or (new_intvar_resolution <= 0):
            raise ValueError(f"{self._intvar_name}_resolution must be a positive float.")
        
        if not isinstance(new_range_intvar, tuple) or (len(new_range_intvar) != 2) or not isinstance(new_range_intvar[0], (float, int, np.number)) or not isinstance(new_range_intvar[1], (float, int, np.number)):
            raise ValueError(f"range_{self._intvar_name} must be a tuple of 2 floats.")
        
        if not isinstance(new_atol, (float, int, np.number)) or (new_atol <= 0):
            raise ValueError("atol must be a positive float.")
        
        new_atol = min(new_atol, new_intvar_resolution)
        changed = [a != b for a, b in zip(
            [new_intvar_resolution, new_range_intvar],
            [self._intvar_resolution, self._range_intvar])]
        
        if any(changed):
            if self._nfdensity is None:
                raise RuntimeError(
                    "Cannot change integration parameters that require recalculating the maps "
                    "without providing the 'nfdensity' model."
                )
            self.clear_cache()
        
        self._intvar_resolution = new_intvar_resolution
        self._range_intvar = new_range_intvar
        self._atol = new_atol
    
    @staticmethod
    def _build_nodes(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, w = np.polynomial.legendre.leggauss(degree)
        return torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)

    @staticmethod
    def _scale_nodes_exp(start: float, stop: float,
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = stop - start

        out_n = (nodes_u + 1).mul(0.5).pow(2).mul(diff).add(start)
        out_w = (nodes_u + 1).mul(0.5).mul(weights_u).mul(diff)

        return out_n, out_w
    
    @staticmethod
    def _scale_nodes_center(start: float, stop: float, center: float,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_left = (nodes_u < 0)
        
        width_left = torch.tensor(center - start, dtype=nodes_u.dtype)
        width_right = torch.tensor(stop - center, dtype=nodes_u.dtype)

        scale = torch.where(mask_left, width_left, width_right)

        out_n = nodes_u.pow(3).mul(scale).add(center)
        out_w = nodes_u.pow(2).mul(3).mul(weights_u).mul(scale)

        return out_n, out_w

    @staticmethod
    def _scale_nodes_left_flank(start: float, stop: float, 
                                nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = stop - start
        v = (nodes_u - 1.0) * 0.5  # Maps [-1, 1] to [-1, 0]
        
        out_n = stop + diff * v.pow(3)
        out_w = weights_u * 1.5 * diff * v.pow(2)
        
        return out_n, out_w

    @staticmethod
    def _scale_nodes_right_flank(start: float, stop: float, 
                                 nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = stop - start
        v = (nodes_u + 1.0) * 0.5  # Maps [-1, 1] to [0, 1]
        
        out_n = start + diff * v.pow(3)
        out_w = weights_u * 1.5 * diff * v.pow(2)
        
        return out_n, out_w

    @staticmethod
    def _subdivide_interval(interval, resolution: float):
        a, b = interval
        diff = b - a

        if diff <= resolution:
            return [a, b]

        num_points = int(np.ceil(diff / resolution)) + 1

        subdivided = np.linspace(a, b, num_points, dtype=float).tolist()

        return subdivided
    
    def _set_norm_dimensions(self, norm_dimensions: str):
        raise ValueError(f"Unsupported normalization dimensions {norm_dimensions}")
    
    @abstractmethod
    def _init_maps(self): ...

    @abstractmethod
    def cache_to_file(self, filename: Union[str, Path]): ...
    
    @abstractmethod
    def cache_from_file(self, filename: Union[str, Path]): ...
    
    @abstractmethod
    def query_normalization(self, **kwargs) -> np.ndarray: ...

    def init_cache(self):
        if self._maps is None:
            self.init_setup()
            self._init_maps()
            if self._norm_dimensions == "Em":
                self._integration_Em()
            elif self._norm_dimensions == "Phi":
                self._integration_Phi()
            else:
                raise ValueError(f"Unsupported normalization dimensions {self._norm_dimensions}")
    
    def _integration_Em(self):
        raise ValueError(f"Unsupported normalization dimensions {self._norm_dimensions}")
    
    @abstractmethod
    def _init_interpolator(self): ...

    def _integration_Phi(self):
        raise ValueError(f"Unsupported normalization dimensions {self._norm_dimensions}")

    def init_setup(self):
        if self._intvar is None:
            self._set_domain(intvar=self._intvar_input, menergy_keV=self._menergy_keV_input, scatt_angle_rad=self._scatt_angle_rad_input)

    def clear_cache(self):
        self._coordinates = None
        self._maps = None
        self._intvar = None
        self._menergy_keV = None
        self._scatt_angle_rad = None

    def _validate_intervals(self, name: str, val: list, allow_float: bool = False, check_overlap: bool = False) -> list:
            if not isinstance(val, list):
                raise TypeError(f"{name} must be a list.")
            
            parsed_items = []
            
            for item in val:
                if isinstance(item, (float, int, np.number)):
                    if not allow_float:
                        raise ValueError(f"Elements of {name} cannot be a float. They must be tuple intervals (e.g., (a, b)).")
                    else:
                        parsed_items.append(float(item))
                
                elif isinstance(item, tuple):
                    if len(item) != 2:
                        raise ValueError(f"Each interval in {name} must have exactly 2 elements. Got {item}.")
                    
                    if all([isinstance(x, (float, int, np.number)) for x in item]):
                        a, b = float(item[0]), float(item[1])
                    elif all([isinstance(x, Time) for x in item]):  
                        a, b = float(item[0].utc.unix), float(item[1].utc.unix)
                    else:
                        raise TypeError(f"Unrecognized elements in {name}: {item}. Expected numbers or times.")
                    
                    if np.isnan(a) or np.isnan(b):
                        if self._range_intvar is None:
                            raise ValueError("Cannot have NaN intervals if the range is not specified.")
                        if np.isnan(a):
                            a = self._range_intvar[0]
                        if np.isnan(b):
                            b = self._range_intvar[1]
                    
                    if a >= b:
                        raise ValueError(f"Intervals in {name} must be strictly increasing (a < b). Got {item}.")
                    
                    parsed_items.append([a, b])
                else:
                    raise TypeError(f"Unrecognized element in {name}: {item}. Expected float/int or tuple.")
            
            if check_overlap:
                intervals_only = [item for item in parsed_items if isinstance(item, list)]
                if len(intervals_only) > 1:
                    sorted_intervals = sorted(intervals_only, key=lambda x: x[0])
                    for i in range(1, len(sorted_intervals)):
                        prev_end = sorted_intervals[i-1][1]
                        curr_start = sorted_intervals[i][0]
                        if curr_start < prev_end:
                            raise ValueError(
                                f"Intervals in {name} cannot overlap. "
                                f"Found overlap between {sorted_intervals[i-1]} and {sorted_intervals[i]}."
                            )

            return parsed_items

    def _set_domain(self,
                    intvar: T_IntVarList,
                    menergy_keV: Optional[MEnergyList] = None,
                    scatt_angle_rad: Optional[MEnergyList] = None,
                    ):
        
        intvar_val = self._validate_intervals(self._intvar_name, intvar, allow_float=self._intvar_allow_float)
        
        # Check intervals
        
        if menergy_keV is not None and scatt_angle_rad is not None and self._norm_dimensions != "EmPhi":
            raise ValueError("Only one of menergy_keV or scatt_angle_rad can be specified given the normalization dimensions.")
        if scatt_angle_rad is None and menergy_keV is None:
            raise ValueError("At least one of menergy_keV or scatt_angle_rad must be specified.")
        if menergy_keV is not None:
            if self._norm_dimensions != "Phi":
                menergy_arr = self._validate_intervals("menergy_keV", menergy_keV, allow_float=False, check_overlap=True)
            else:
                raise ValueError("menergy_keV is not supported for normalization dimensions Phi.")
        if scatt_angle_rad is not None:
            if self._norm_dimensions != "Em":
                scatt_arr = self._validate_intervals("scatt_angle_rad", scatt_angle_rad, allow_float=False, check_overlap=True)
            else:
                raise ValueError("scatt_angle_rad is not supported for normalization dimensions Em.")
        if menergy_keV is not None and scatt_angle_rad is not None:
            num_menergy = len(menergy_arr)
            num_scatt = len(scatt_arr)
        
            if num_menergy > 1 and num_scatt > 1:
                if num_menergy != num_scatt:
                    raise ValueError(
                        f"Interval mismatch: menergy_keV has {num_menergy} intervals, "
                        f"but scatt_angle_rad has {num_scatt}."
                    )
        
        # Check if resolutions are specified
        
        if any([isinstance(elem, list) for elem in intvar_val]) and self._intvar_resolution is None:
            raise ValueError(f"If {self._intvar_name} contains intervals, {self._intvar_name}_resolution must be specified.")

        if (self._norm_dimensions == "Em") and self._menergy_keV_resolution is None:
            raise ValueError("Normalization dimensions Em requires menergy_keV_resolution to be specified.")
        
        # Set the arrays

        self._intvar = intvar_val
        self._menergy_keV = menergy_arr if menergy_keV is not None else None
        self._scatt_angle_rad = scatt_arr if scatt_angle_rad is not None else None

    def _inference_helper(self, context: torch.Tensor, source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        active_pool = True
        if isinstance(self._nfdensity, NFBase):
            active_pool = self._nfdensity.active_pool
            if not active_pool:
                self._nfdensity.init_compute_pool()
        
        weighted_density = self._nfdensity.evaluate_density(context, source).to(torch.float64) * weights
        
        if not active_pool:
            self._nfdensity.shutdown_compute_pool()
        
        return weighted_density

class NFNormalizationMapEmBase(NFNormalizationMapBase[T_IntVarList, T_NormDensityType]):
    
    @property
    def menergy_keV_resolution(self):
        return self._menergy_keV_resolution
    @menergy_keV_resolution.setter
    def menergy_keV_resolution(self, val: float): self._set_integration_parameters(menergy_keV_resolution=val)
    
    def _set_integration_parameters(self,
                                   intvar_resolution: Optional[float] = -1.0,
                                   menergy_keV_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[Tuple[float, float]] = None,
                                   atol: Optional[float] = -1.0):
        
        super()._set_integration_parameters(intvar_resolution=intvar_resolution,
                                            range_intvar=range_intvar,
                                            atol=atol)
        
        new_menergy_keV_resolution = menergy_keV_resolution if menergy_keV_resolution != -1.0 else self._menergy_keV_resolution
        
        if not isinstance(new_menergy_keV_resolution, (float, int, np.number)) or (new_menergy_keV_resolution <= 0):
            raise ValueError("menergy_keV_resolution must be a positive float.")
        
        changed = [a != b for a, b in zip(
            [new_menergy_keV_resolution, ],
            [self._menergy_keV_resolution, ])]
        
        if any(changed):
            if self._nfdensity is None:
                raise RuntimeError(
                    "Cannot change integration parameters that require recalculating the maps "
                    "without providing the 'nfdensity' model."
                )
            self.clear_cache()
        
        self._menergy_keV_resolution = new_menergy_keV_resolution
    
    def _set_norm_dimensions(self, norm_dimensions: str):
        if norm_dimensions not in ["Em"]:
            raise ValueError(f"Unsupported normalization dimensions {norm_dimensions}")
        self._norm_dimensions = norm_dimensions
    
    def _integration_Em(self):
        raise NotImplementedError
    
    def _check_peak_active(self, true_center: float, width: float, base_nodes: int, menergy_keV: List[float]) -> Tuple[float, float, int]:
        Emin, Emax = menergy_keV
        
        if (true_center - width <= Emax) and (true_center + width >= Emin):
            return true_center, width, base_nodes
            
        return np.nan, np.nan, 0
    
    def _fill_Em_bkg(self, E1, E2):
            diff = E2 - E1
            return 2 if diff < self._menergy_keV_resolution else int(np.ceil(diff / self._menergy_keV_resolution))
    
    def _node_logic_Em(self, peaks: List[float], widths: List[Tuple[float, int]], menergy_keV: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        Emin, Emax = menergy_keV
        n_peaks = len(peaks)
        
        nodes, weights = [], []
        El = Emin
        
        for i in range(n_peaks):
            EC = peaks[i]
            W = widths[i][0]
            N_base_flank = max(2, widths[i][1] // 2) 
            
            # A. Background space before this peak
            Start_L = np.clip(EC - W, a_min=El, a_max=None)
            
            if Start_L > El:
                n_nodes = self._fill_Em_bkg(El, Start_L)
                n, w = self._build_nodes(n_nodes)
                n, w = self._scale_nodes_exp(El, Start_L, n, w)
                nodes.append(n)
                weights.append(w)
                El = Start_L
            
            # B. The Left Flank
            End_L = min(EC, Emax)
            
            if Start_L < End_L:
                fraction = (End_L - Start_L) / W
                n_nodes = max(2, int(np.ceil(N_base_flank * fraction)))
                
                n, w = self._build_nodes(n_nodes)
                n, w = self._scale_nodes_left_flank(Start_L, End_L, n, w)
                nodes.append(n)
                weights.append(w)
                El = End_L
                
            # C. The Right Flank
            Er_limit = Emax if i == n_peaks - 1 else (EC + peaks[i+1]) / 2.0
            
            Start_R = max(El, EC) 
            End_R = min(EC + W, Er_limit, Emax)
            
            if Start_R < End_R:
                fraction = (End_R - Start_R) / W
                n_nodes = max(2, int(np.ceil(N_base_flank * fraction)))
                
                n, w = self._build_nodes(n_nodes)
                n, w = self._scale_nodes_right_flank(Start_R, End_R, n, w)
                nodes.append(n)
                weights.append(w)
                El = End_R
            
        # D. Trailing background after all peaks are processed
        if El < Emax:
            n_nodes = self._fill_Em_bkg(El, Emax)
            n, w = self._build_nodes(n_nodes)
            n, w = self._scale_nodes_exp(El, Emax, n, w)
            nodes.append(n)
            weights.append(w)
            
        return (torch.cat(nodes), torch.cat(weights)) if nodes else (torch.empty(0), torch.empty(0))
    
    #def _node_logic_Em(self, peaks: List[float], widths: List[Tuple[float, int]], menergy_keV: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    #    Emin, Emax = menergy_keV
    #    n_peaks = len(peaks)
    #    
    #    def n_fill_peaks(E1, E2):
    #        diff = E2 - E1
    #        if diff < self._menergy_keV_resolution:
    #            return 2
    #        else:
    #            return int(np.ceil(diff / self._menergy_keV_resolution))
    #    
    #    if n_peaks == 0:
    #        n_nodes = n_fill_peaks(Emin, Emax)
    #        n, w = self._build_nodes(n_nodes)
    #        return self._scale_nodes_exp(Emin, Emax, n, w)
    #    else:
    #        nodes = []
    #        weights = []
    #        
    #        diffs = [widths[i][0] for i in range(n_peaks)]
    #        n_nodes_peaks = [widths[i][1] for i in range(n_peaks)]
    #        
    #        El = Emin
    #        for i in range(n_peaks):
    #            EC = peaks[i]
    #            E1 = np.clip(EC - diffs[i], a_min=El, a_max=None)
    #            Er = Emax if i == n_peaks - 1 else (EC + peaks[i+1])/2
    #            E2 = np.clip(EC + diffs[i], a_min=None, a_max=Er)
    #            
    #            if not np.isclose(El, E1):
    #                n_nodes = n_fill_peaks(El, E1)
    #                n, w = self._build_nodes(n_nodes)
    #                n, w = self._scale_nodes_exp(El, E1, n, w)
    #                nodes.append(n)
    #                weights.append(w)
    #            
    #            n_nodes = n_nodes_peaks[i]
    #            n, w = self._build_nodes(n_nodes)
    #            n, w = self._scale_nodes_center(E1, E2, EC, n, w)
    #            nodes.append(n)
    #            weights.append(w)
    #            
    #            El = E2
    #        
    #        if not np.isclose(El, Emax):
    #            n_nodes = n_fill_peaks(El, Emax)
    #            n, w = self._build_nodes(n_nodes)
    #            n, w = self._scale_nodes_exp(El, Emax, n, w)
    #            nodes.append(n)
    #            weights.append(w)
    #        
    #        return torch.cat(nodes), torch.cat(weights)

    def _nodes_Em(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def _Em_peak_params(self, **kwargs) -> Tuple[float, int]: ...

    def _tensor_logic_Em(self): ...
