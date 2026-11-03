import torch
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity
from typing import Union, Optional, Tuple, List, Sequence, TypeVar, Generic
from pathlib import Path
from abc import ABC, abstractmethod
import copy
from contextlib import contextmanager
from functools import lru_cache

NumberTypes = Union[float, int, np.number]
Interval = Tuple[NumberTypes, NumberTypes]
ArrayLike = Union[np.ndarray, torch.Tensor, Sequence, NumberTypes]
RangeLike = Union[Tuple[NumberTypes, NumberTypes], Quantity, Time]
 
T_IntVarType = TypeVar('T_IntVarType')
T_NormDensityType = TypeVar('T_NormDensityType')

T_IntVarType = TypeVar('T_IntVarType')
T_NormDensityType = TypeVar('T_NormDensityType')

from .NFBase import NFBase

class NFNormalizationMapBase(ABC, Generic[T_IntVarType, T_NormDensityType]):
    
    # Need to be defined by implementation
    _intvar_allow_float: bool       
    _intvar_name: str
    _intvar_unit: Optional[u.UnitBase] 
    
    _menergy_unit: u.UnitBase = u.keV
    _scatt_angle_unit: u.UnitBase = u.rad
    
    _range_intvar: Tuple[float, float]
    _range_menergy_keV: Optional[Tuple[float, float]]
    _range_scatt_angle_rad: Optional[Tuple[float, float]]
    
    def __init__(self,
                 nfdensity: Optional[T_NormDensityType] = None,
                 intvar: Optional[T_IntVarType] = None,
                 menergy: Optional[Quantity] = None,
                 scatt_angle: Optional[Quantity] = None,
                 ):
        
        self._nfdensity = nfdensity
        self._intvar_input = copy.deepcopy(intvar)
        self._menergy_input = copy.deepcopy(menergy)
        self._scatt_angle_input = copy.deepcopy(scatt_angle)
        
        self._coordinates = None
        self._maps = None
        self._interpolator = None
        
        self._intvar: Optional[np.ndarray] = None
        self._menergy_keV: Optional[np.ndarray] = None
        self._scatt_angle_rad: Optional[np.ndarray] = None
        
        self._range_intvar: Optional[Tuple[float, float]] = None
        self._range_menergy_keV: Optional[Tuple[float, float]] = None
        self._range_scatt_angle_rad: Optional[Tuple[float, float]] = None
        self._intvar_resolution: Optional[float] = None
        self._atol: float = 1e-3
        self._menergy_keV_resolution: Optional[float] = None
        self._is_identity_cut: bool = False
        
        if nfdensity is not None:
            self._set_norm_dimensions(nfdensity.norm_dimensions)
        else:
            self._norm_dimensions = None
    
    @classmethod
    def from_cache(cls, filename: Union[str, Path]) -> "NFNormalizationMapBase":
        instance = cls()
        instance.cache_from_file(filename)
        return instance
    
    @property
    def atol(self):
        return self._atol
    @atol.setter
    def atol(self, val: float): self.set_integration_parameters(atol=val)
    
    @property
    def norm_dimensions(self):
        return self._norm_dimensions
    
    @property
    def intvar(self):
        return copy.deepcopy(self._intvar_input)
    @intvar.setter
    def intvar(self, val: T_IntVarType): self._set_inputs(intvar=val)
    
    @property
    def menergy(self) -> Optional[Quantity]:
        return copy.deepcopy(self._menergy_input)
    @menergy.setter
    def menergy(self, val: Optional[Quantity]): self._set_inputs(menergy=val)
    
    @property
    def scatt_angle(self) -> Optional[Quantity]:
        return copy.deepcopy(self._scatt_angle_input)
    @scatt_angle.setter
    def scatt_angle(self, val: Optional[Quantity]): self._set_inputs(scatt_angle=val)
    
    def _set_inputs(self, **new_inputs):
        self._require_density_for_change()
        
        inputs = {"intvar": self._intvar_input,
                  "menergy": self._menergy_input,
                  "scatt_angle": self._scatt_angle_input}
        inputs.update({k: copy.deepcopy(v) for k, v in new_inputs.items()})
        
        self._validate_domain(**inputs) 
        
        if inputs["intvar"] != self._intvar_input:
            self._intvar_input = inputs["intvar"]
            clear_cache_needed = True
        if inputs["menergy"] != self._menergy_input:
            self._menergy_input = inputs["menergy"]
            clear_cache_needed = True
        if inputs["scatt_angle"] != self._scatt_angle_input:
            self._scatt_angle_input = inputs["scatt_angle"]
            clear_cache_needed = True
        if clear_cache_needed:
            self.clear_cache()
    
    def _require_density_for_change(self):
        if self._nfdensity is None:
            raise RuntimeError(
                "Cannot change parameters that require recalculating the maps "
                "without providing the 'nfdensity' model."
            )
    
    def _parse_range(self, name: str, val: Optional[RangeLike], unit: Optional[u.UnitBase]) -> Optional[Tuple[float, float]]:
        if val is None:
            return None
        
        if isinstance(val, (tuple, list)) and len(val) == 2:
            if all(isinstance(x, Quantity) for x in val):
                val = Quantity(val)
            elif all(isinstance(x, Time) for x in val):
                val = Time(list(val))
        
        if isinstance(val, Time):
            if unit is not None:
                raise TypeError(f"{name} must be given as a Quantity (or plain numbers in {unit}), not as Time.")
            arr = val.utc.unix
        elif isinstance(val, Quantity):
            if unit is None:
                raise TypeError(f"{name} must be given as Time (or plain unix seconds), not as a Quantity.")
            try:
                arr = val.to_value(unit)
            except u.UnitsError as err:
                raise ValueError(f"{name} has incompatible units: {err}") from err
        else:
            arr = val
        
        try:
            arr = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError) as err:
            raise ValueError(f"{name} must be a (min, max) pair of numbers.") from err
        
        if arr.shape != (2,) or np.isnan(arr).any() or not arr[0] < arr[1]:
            raise ValueError(f"{name} must be a (min, max) pair with min < max, got {val}.")
        return (float(arr[0]), float(arr[1]))
    
    def set_integration_parameters(self,
                                   intvar_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[RangeLike] = None,
                                   range_menergy_keV: Optional[RangeLike] = None,
                                   range_scatt_angle_rad: Optional[RangeLike] = None,
                                   atol: Optional[float] = -1.0):
        
        new_resolution = self._intvar_resolution if intvar_resolution == -1.0 else intvar_resolution
        new_atol = self._atol if atol == -1.0 else atol
        new_ranges = (
            self._parse_range("range_intvar", range_intvar, self._intvar_unit) or self._range_intvar,
            self._parse_range("range_menergy_keV", range_menergy_keV, self._menergy_unit) or self._range_menergy_keV,
            self._parse_range("range_scatt_angle_rad", range_scatt_angle_rad, self._scatt_angle_unit) or self._range_scatt_angle_rad,
        )
        
        if new_ranges[0] is None:
            raise ValueError("range_intvar must be defined.")
        
        if new_resolution is not None and (not isinstance(new_resolution, (float, int, np.number)) or new_resolution <= 0):
            raise ValueError(f"{self._intvar_name}_resolution must be a positive float.")
        
        if not isinstance(new_atol, (float, int, np.number)) or new_atol <= 0:
            raise ValueError("atol must be a positive float.")
        if new_resolution is not None:
            new_atol = min(new_atol, new_resolution)
        
        old_ranges = (self._range_intvar, self._range_menergy_keV, self._range_scatt_angle_rad)
        ranges_changed = new_ranges != old_ranges
        changed = ranges_changed or new_resolution != self._intvar_resolution
        
        if changed:
            self._require_density_for_change()
        
        old_state = (self._intvar_resolution, self._atol, *old_ranges)
        (self._range_intvar, self._range_menergy_keV, self._range_scatt_angle_rad) = new_ranges
        self._intvar_resolution, self._atol = new_resolution, new_atol
        
        has_input = self._menergy_input is not None or self._scatt_angle_input is not None
        if ranges_changed and has_input:
            try:
                self._validate_domain(self._intvar_input, self._menergy_input, self._scatt_angle_input)
            except Exception:
                (self._intvar_resolution, self._atol,
                 self._range_intvar, self._range_menergy_keV, self._range_scatt_angle_rad) = old_state
                raise
        
        if changed:
            self.clear_cache()
    
    @staticmethod
    @lru_cache(maxsize=512)
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
    def _subdivide_interval(interval, resolution: float) -> list:
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
            self._set_domain(intvar=self._intvar_input, menergy=self._menergy_input, scatt_angle=self._scatt_angle_input)

    def clear_cache(self):
        self._coordinates = None
        self._maps = None
        self._interpolator = None
        self._intvar = None
        self._menergy_keV = None
        self._scatt_angle_rad = None
        self._is_identity_cut = False

    def _intervals_to_array(self, name: str, val, unit: Optional[u.UnitBase]) -> np.ndarray:
        if unit is None:
            if not isinstance(val, Time):
                raise TypeError(f"{name} must be an astropy Time of shape (N, 2), got {type(val).__name__}.")
            arr = val.utc.unix
        else:
            if not isinstance(val, Quantity):
                raise TypeError(f"{name} must be an astropy Quantity of shape (N, 2) with unit convertible to "
                                f"{unit}, got {type(val).__name__}.")
            try:
                arr = val.to_value(unit)
            except u.UnitsError as err:
                raise ValueError(f"{name} has incompatible units: {err}") from err
        
        arr = np.array(arr, dtype=np.float64) 
        if arr.shape == (2,):
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
            raise ValueError(f"{name} must have shape (N, 2) with N >= 1, got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite. Provide explicit numeric bounds.")
        return arr

    def _check_intervals(self, name: str, arr: np.ndarray, unit: Optional[u.UnitBase],
                         range_bounds: Tuple[float, float],
                         allow_point: bool = False, check_overlap: bool = True,
                         clamp: bool = True) -> np.ndarray:
        unit_str = "unix s (UTC)" if unit is None else str(unit)
        lo, hi = range_bounds
        
        arr[:] = np.where(np.isclose(arr, lo, rtol=1e-6), lo, arr)
        arr[:] = np.where(np.isclose(arr, hi, rtol=1e-6), hi, arr)
        a, b = arr[:, 0], arr[:, 1]  # views
        
        is_point = np.isclose(a, b, rtol=1e-6)
        if is_point.any():
            if not allow_point:
                raise ValueError(f"Intervals in {name} must have a < b, found zero-width intervals "
                                 f"{arr[is_point].tolist()} ({unit_str}).")
            arr[is_point] = arr[is_point].mean(axis=1, keepdims=True)
        
        reversed_ = ~is_point & (a > b)
        if reversed_.any():
            raise ValueError(f"Intervals in {name} must be strictly increasing (a < b). "
                             f"Got {arr[reversed_].tolist()} ({unit_str}).")
        
        outside = (a < lo) | (b > hi)
        if clamp:
            if (outside & is_point).any():
                raise ValueError(f"Value(s) {arr[outside & is_point][:, 0].tolist()} in {name} are outside the "
                                 f"valid range {range_bounds} ({unit_str}).")
            a_c, b_c = np.maximum(a, lo), np.minimum(b, hi)
            empty = ~is_point & (a_c >= b_c)
            if empty.any():
                raise ValueError(f"Interval(s) {arr[empty].tolist()} in {name} lie entirely outside the valid range "
                                 f"{range_bounds} ({unit_str}); clamping would produce an empty interval.")
            arr[:, 0], arr[:, 1] = a_c, b_c
        elif outside.any():
            raise ValueError(f"Interval(s) {arr[outside].tolist()} in {name} are outside the valid range "
                             f"{range_bounds} ({unit_str}).")
        
        if check_overlap and len(arr) > 1:
            order = np.lexsort((arr[:, 1], arr[:, 0]))
            s, e, p = arr[order, 0], arr[order, 1], is_point[order]
            prev_end = np.maximum.accumulate(e)[:-1]
            overlap = (s[1:] < prev_end) | ((s[1:] <= prev_end) & (p[1:] | p[:-1]))
            if overlap.any():
                i = int(np.argmax(overlap)) + 1
                j = int(np.argmax(e[:i] >= s[i]))
                raise ValueError(f"Entries in {name} cannot overlap. Found overlap between "
                                 f"{arr[order[j]].tolist()} and {arr[order[i]].tolist()} ({unit_str}).")
        
        return arr

    def _validate_intervals(self, name: str, val, unit: Optional[u.UnitBase],
                            range_bounds: Tuple[float, float], **kwargs) -> np.ndarray:
        return self._check_intervals(name, self._intervals_to_array(name, val, unit), unit, range_bounds, **kwargs)

    def _validate_domain(self,
                         intvar: Optional[T_IntVarType],
                         menergy: Optional[Quantity] = None,
                         scatt_angle: Optional[Quantity] = None,
                         ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool]:
        
        if self._range_intvar is None:
            raise RuntimeError(
                f"{type(self).__name__} must define _range_intvar (a (min, max) tuple) "
                f"in its __init__ before the domain can be set."
            )
        
        if menergy is not None and scatt_angle is not None and self._norm_dimensions != "EmPhi":
            raise ValueError("Only one of menergy or scatt_angle can be specified given the normalization dimensions.")
        if menergy is None and scatt_angle is None:
            raise ValueError("At least one of menergy or scatt_angle must be specified.")
        if menergy is not None and self._norm_dimensions == "Phi":
            raise ValueError("menergy is not supported for normalization dimensions Phi.")
        if scatt_angle is not None and self._norm_dimensions == "Em":
            raise ValueError("scatt_angle is not supported for normalization dimensions Em.")
        
        
        if intvar is None:
            if not (np.isfinite(self._range_intvar[0]) and np.isfinite(self._range_intvar[1])):
                raise ValueError(
                    f"No {self._intvar_name} was provided, and _range_intvar={self._range_intvar} "
                    f"is not finite. Either provide {self._intvar_name} explicitly, or set a finite "
                    f"range_intvar to be able to default to the full range."
                )
            intvar_arr = self._check_intervals(self._intvar_name, np.array([self._range_intvar], dtype=np.float64),
                                               self._intvar_unit, self._range_intvar,
                                               allow_point=self._intvar_allow_float, clamp=False)
        else:
            intvar_arr = self._validate_intervals(self._intvar_name, intvar, self._intvar_unit, self._range_intvar,
                                                  allow_point=self._intvar_allow_float, clamp=False)
        
        menergy_arr, scatt_arr = None, None
        
        if menergy is not None:
            if self._range_menergy_keV is None:
                raise RuntimeError(
                    f"{type(self).__name__} must define _range_menergy_keV (a fixed (min, max) tuple) "
                    f"in its __init__ before menergy can be used."
                )
            menergy_arr = self._validate_intervals("menergy", menergy, self._menergy_unit, self._range_menergy_keV,
                                                   clamp=True)
        
        if scatt_angle is not None:
            if self._range_scatt_angle_rad is None:
                raise RuntimeError(
                    f"{type(self).__name__} must define _range_scatt_angle_rad (a fixed (min, max) tuple) "
                    f"in its __init__ before scatt_angle can be used."
                )
            scatt_arr = self._validate_intervals("scatt_angle", scatt_angle, self._scatt_angle_unit,
                                                 self._range_scatt_angle_rad, clamp=True)
        
        if menergy_arr is not None and scatt_arr is not None:
            n_menergy, n_scatt = len(menergy_arr), len(scatt_arr)
            if n_menergy > 1 and n_scatt > 1 and n_menergy != n_scatt:
                raise ValueError(
                    f"Interval mismatch: menergy has {n_menergy} intervals, but scatt_angle has {n_scatt}."
                )
        
        if (intvar_arr[:, 0] != intvar_arr[:, 1]).any() and self._intvar_resolution is None:
            raise ValueError(f"If {self._intvar_name} contains intervals, {self._intvar_name}_resolution must be specified.")

        if (self._norm_dimensions == "Em") and self._menergy_keV_resolution is None:
            raise ValueError("Normalization dimensions Em requires menergy_keV_resolution to be specified.")
        
        is_identity_cut = (
            (menergy_arr is None or self._is_full_domain(menergy_arr, self._range_menergy_keV))
            and (scatt_arr is None or self._is_full_domain(scatt_arr, self._range_scatt_angle_rad))
        )
        
        return intvar_arr, menergy_arr, scatt_arr, is_identity_cut
    
    @staticmethod
    def _is_full_domain(intervals: Optional[np.ndarray], domain_range: Optional[Tuple[float, float]]) -> bool:
        if domain_range is None or intervals is None or len(intervals) != 1:
            return False
        a, b = intervals[0]
        return bool(np.isclose(a, domain_range[0], rtol=1e-6) and np.isclose(b, domain_range[1], rtol=1e-6))

    def _set_domain(self,
                    intvar: Optional[T_IntVarType],
                    menergy: Optional[Quantity] = None,
                    scatt_angle: Optional[Quantity] = None,
                    ):
        (self._intvar, self._menergy_keV, self._scatt_angle_rad, 
         self._is_identity_cut) = self._validate_domain(intvar, menergy, scatt_angle)

    def _cut_matches(self, name: str, own: Optional[np.ndarray], other: Optional[Quantity],
                     unit: u.UnitBase, bounds: Optional[Tuple[float, float]]) -> bool:
        if own is None or other is None:
            return own is None and other is None
        try:
            other = self._validate_intervals(name, other, unit, bounds, clamp=True)
        except (TypeError, ValueError):
            return False
        if other.shape != own.shape:
            return False
        
        def _sorted(x): return x[np.lexsort((x[:, 1], x[:, 0]))]
        return bool(np.all(np.isclose(_sorted(other), _sorted(own), rtol=1e-8)))
    
    def check_compatibility(self, menergy: Optional[Quantity] = None, scatt_angle: Optional[Quantity] = None) -> bool:
        self.init_setup()
        return (self._cut_matches("menergy", self._menergy_keV, menergy,
                                  self._menergy_unit, self._range_menergy_keV)
                and self._cut_matches("scatt_angle", self._scatt_angle_rad, scatt_angle,
                                      self._scatt_angle_unit, self._range_scatt_angle_rad))

    @property
    def _intvar_is_point(self) -> np.ndarray:
        return self._intvar[:, 0] == self._intvar[:, 1]

    @contextmanager
    def _compute_pool(self):
        """Keeps the compute pool of the density alive for the whole block (if it is not already running),
        so that batched evaluation does not start and stop the pool for every batch."""
        started = isinstance(self._nfdensity, NFBase) and not self._nfdensity.active_pool
        if started:
            self._nfdensity.init_compute_pool()
        try:
            yield
        finally:
            if started:
                self._nfdensity.shutdown_compute_pool()

    def _inference_helper(self, context: torch.Tensor, source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        with self._compute_pool():
            return self._nfdensity.evaluate_density(context, source).to(torch.float64) * weights

    @staticmethod
    def _export_intervals(arr: np.ndarray, unit: Optional[u.UnitBase]):
        if unit is None:
            return Time(arr, format="unix", scale="utc")
        return Quantity(arr, unit)

class NFNormalizationMapEmBase(NFNormalizationMapBase[T_IntVarType, T_NormDensityType]):
    
    @property
    def menergy_keV_resolution(self):
        return self._menergy_keV_resolution
    @menergy_keV_resolution.setter
    def menergy_keV_resolution(self, val: float): self.set_integration_parameters(menergy_keV_resolution=val)
    
    def set_integration_parameters(self,
                                   intvar_resolution: Optional[float] = -1.0,
                                   menergy_keV_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[RangeLike] = None,
                                   range_menergy_keV: Optional[RangeLike] = None,
                                   range_scatt_angle_rad: Optional[RangeLike] = None,
                                   atol: Optional[float] = -1.0):
        
        new_menergy_keV_resolution = self._menergy_keV_resolution if menergy_keV_resolution == -1.0 else menergy_keV_resolution
        
        if new_menergy_keV_resolution is not None and (
                not isinstance(new_menergy_keV_resolution, (float, int, np.number)) or new_menergy_keV_resolution <= 0):
            raise ValueError("menergy_keV_resolution must be a positive float.")
        
        changed = new_menergy_keV_resolution != self._menergy_keV_resolution
        if changed:
            self._require_density_for_change()
        
        # Set before calling super(), so that its validation of the domain already sees the new value.
        old_resolution = self._menergy_keV_resolution
        self._menergy_keV_resolution = new_menergy_keV_resolution
        try:
            super().set_integration_parameters(intvar_resolution=intvar_resolution,
                                               range_intvar=range_intvar,
                                               range_menergy_keV=range_menergy_keV,
                                               range_scatt_angle_rad=range_scatt_angle_rad,
                                               atol=atol)
        except Exception:
            self._menergy_keV_resolution = old_resolution
            raise
        
        if changed:
            self.clear_cache()
    
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

    def _nodes_Em(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def _Em_peak_params(self, **kwargs) -> Tuple[float, int]: ...

    def _tensor_logic_Em(self): ...