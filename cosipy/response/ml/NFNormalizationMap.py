import torch
import numpy as np
import healpy as hp
from typing import Union, Optional, Tuple, List, Sequence
import json
import os
import h5py
from pathlib import Path

Interval = Tuple[Union[float, int], Union[float, int]]
ArrayLike = Union[np.ndarray, torch.Tensor, Sequence, float, int, np.number]
IEnergyList = List[Union[float, int, Interval]]
MEnergyList = List[Interval]

from .NFNormalizationDensity import NFNormalizationDensity

# TODO: Shift from response only to response + background? -> Would require different integration of background
# TODO: Add option for ARM
# TODO: Add option for more complex functional relationships

class NFNormalizationMap:
    def __init__(self,
                 nfdensity: Optional[NFNormalizationDensity] = None, 
                 ienergy_keV: Optional[IEnergyList] = None,
                 menergy_keV: Optional[MEnergyList] = None,
                 scatt_angle_rad: Optional[MEnergyList] = None,
                 ):
        self._nfdensity = nfdensity
        self._ienergy_keV_input = ienergy_keV
        self._menergy_keV_input = menergy_keV
        self._scatt_angle_rad_input = scatt_angle_rad 
        
        self._map_nside: int = 32
        self._map_npix: int = hp.nside2npix(self._map_nside)
        self._ienergy_keV_resolution: Optional[float] = 1.5
        self._menergy_keV_resolution: Optional[float] = 8.0
        self._range_energy_keV: Optional[Tuple[float, float]] = (100., 10_000.)
        self._atol: float = 0.1
        self._Em_peak_widths: Optional[Tuple[float, float, float]] = (0.85, 0.70, 0.75)
        self._Em_peak_nums: Optional[Tuple[int, int, int]] = (18, 12, 12)
        
        self._coordinates = None
        self._maps = None
        self._ienergy_keV = None
        self._menergy_keV = None
        self._scatt_angle_rad = None
        
        if nfdensity is not None:
            self._set_norm_dimensions(nfdensity.norm_dimensions)
        else:
            self._norm_dimensions = None
    
    @classmethod
    def from_cache(cls, filename: Union[str, Path]) -> 'NFNormalizationMap':
        instance = cls(nfdensity=None, ienergy_keV=[])
        instance.cache_from_file(filename)
        return instance
    
    @property
    def map_nside(self):
        return self._map_nside
    @map_nside.setter
    def map_nside(self, val: int): self.set_integration_parameters(map_nside=val) 
    
    @property
    def range_energy_keV(self):
        return self._range_energy_keV
    @range_energy_keV.setter
    def range_energy_keV(self, val: Tuple[float, float]): self.set_integration_parameters(range_energy_keV=val)
    
    @property
    def ienergy_keV_resolution(self):
        return self._ienergy_keV_resolution
    @ienergy_keV_resolution.setter
    def ienergy_keV_resolution(self, val: float): self.set_integration_parameters(ienergy_keV_resolution=val)
    
    @property
    def menergy_keV_resolution(self):
        return self._menergy_keV_resolution
    @menergy_keV_resolution.setter
    def menergy_keV_resolution(self, val: float): self.set_integration_parameters(menergy_keV_resolution=val)
    
    @property
    def atol(self):
        return self._atol
    @atol.setter
    def atol(self, val: float): self.set_integration_parameters(atol=val)
    
    @property
    def Em_peak_widths(self):
        return self._Em_peak_widths
    @Em_peak_widths.setter
    def Em_peak_widths(self, val: Tuple[float, float, float]): self.set_integration_parameters(Em_peak_widths=val)
    
    @property
    def Em_peak_nums(self):
        return self._Em_peak_nums
    @Em_peak_nums.setter
    def Em_peak_nums(self, val: Tuple[int, int, int]): self.set_integration_parameters(Em_peak_nums=val)
    
    @property
    def norm_dimensions(self):
        return self._norm_dimensions
    
    def clear_cache(self):
        self._coordinates = None
        self._maps = None
        self._ienergy_keV = None
        self._menergy_keV = None
        self._scatt_angle_rad = None
    
    def set_integration_parameters(self,
                                   map_nside: Optional[int] = -1,
                                   ienergy_keV_resolution: Optional[float] = -1.0,
                                   menergy_keV_resolution: Optional[float] = -1.0,
                                   range_energy_keV: Optional[Tuple[float, float]] = None,
                                   atol: Optional[float] = -1.0,
                                   Em_peak_widths: Optional[Tuple[float, float, float]] = None,
                                   Em_peak_nums: Optional[Tuple[int, int, int]] = None):
        
        new_map_nside = map_nside if map_nside != -1 else self._map_nside
        new_ienergy_keV_resolution = ienergy_keV_resolution if ienergy_keV_resolution != -1.0 else self._ienergy_keV_resolution
        new_menergy_keV_resolution = menergy_keV_resolution if menergy_keV_resolution != -1.0 else self._menergy_keV_resolution
        new_range_energy_keV = range_energy_keV or self._range_energy_keV
        new_atol = atol if atol != -1.0 else self._atol
        new_Em_peak_widths = Em_peak_widths or self._Em_peak_widths
        new_Em_peak_nums = Em_peak_nums or self._Em_peak_nums
        
        if not isinstance(new_map_nside, (int, np.integer)) or (new_map_nside <= 0):
            raise ValueError("map_nside must be a positive integer.")
        
        if not isinstance(new_ienergy_keV_resolution, (float, int, np.number)) or (new_ienergy_keV_resolution <= 0):
            raise ValueError("ienergy_keV_resolution must be a positive float.")
        
        if not isinstance(new_menergy_keV_resolution, (float, int, np.number)) or (new_menergy_keV_resolution <= 0):
            raise ValueError("menergy_keV_resolution must be a positive float.")
        
        if not isinstance(new_range_energy_keV, tuple) or (len(new_range_energy_keV) != 2) or not isinstance(new_range_energy_keV[0], (float, int, np.number)) or not isinstance(new_range_energy_keV[1], (float, int, np.number)):
            raise ValueError("range_energy_keV must be a tuple of 2 floats.")
        
        if not isinstance(new_atol, (float, int, np.number)) or (new_atol <= 0):
            raise ValueError("atol must be a positive float.")
        
        if not isinstance(new_Em_peak_widths, tuple) or (len(new_Em_peak_widths) != 3) or not isinstance(new_Em_peak_widths[0], (float, int, np.number)) or not isinstance(new_Em_peak_widths[1], (float, int, np.number)) or not isinstance(new_Em_peak_widths[2], (float, int, np.number)):
            raise ValueError("Em_peak_widths must be a tuple of 3 floats.")
        
        if not isinstance(new_Em_peak_nums, tuple) or (len(new_Em_peak_nums) != 3) or not isinstance(new_Em_peak_nums[0], (int, np.integer)) or not isinstance(new_Em_peak_nums[1], (int, np.integer)) or not isinstance(new_Em_peak_nums[2], (int, np.integer)):
            raise ValueError("Em_peak_nums must be a tuple of 3 integers.")
        
        new_atol = min(new_atol, new_ienergy_keV_resolution)
        changed = [a != b for a, b in zip(
            [new_map_nside, new_ienergy_keV_resolution, new_menergy_keV_resolution, new_range_energy_keV, new_Em_peak_widths, new_Em_peak_nums],
            [self._map_nside, self._ienergy_keV_resolution, self._menergy_keV_resolution, self._range_energy_keV, self._Em_peak_widths, self._Em_peak_nums])]
        
        if any(changed):
            if self._nfdensity is None:
                raise RuntimeError(
                    "Cannot change integration parameters that require recalculating the maps "
                    "without providing the 'nfdensity' model."
                )
            self.clear_cache()
        
        self._map_nside = new_map_nside
        self._ienergy_keV_resolution = new_ienergy_keV_resolution
        self._menergy_keV_resolution = new_menergy_keV_resolution
        self._range_energy_keV = new_range_energy_keV
        self._atol = new_atol
        self._Em_peak_nums = new_Em_peak_nums
        self._Em_peak_widths = new_Em_peak_widths
                       
    def _set_norm_dimensions(self, norm_dimensions: str):
        if norm_dimensions not in ["Em"]: # Currently only "Em" is supported
            raise ValueError(f"Unsupported normalization dimensions {norm_dimensions}")
        self._norm_dimensions = norm_dimensions
    
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
                    
                    a, b = float(item[0]), float(item[1])
                    
                    if np.isnan(a) or np.isnan(b):
                        if self._range_energy_keV is None:
                            raise ValueError("Cannot have NaN intervals if range_energy_keV is not specified.")
                        if np.isnan(a):
                            a = self._range_energy_keV[0]
                        if np.isnan(b):
                            b = self._range_energy_keV[1]
                    
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
                    ienergy_keV: IEnergyList,
                    menergy_keV: Optional[MEnergyList] = None,
                    scatt_angle_rad: Optional[MEnergyList] = None,
                    ):
        
        ienergy_val = self._validate_intervals("ienergy_keV", ienergy_keV, allow_float=True)
        
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
        
        if any([isinstance(elem, list) for elem in ienergy_val]) and self._ienergy_keV_resolution is None:
            raise ValueError("If ienergy_keV contains intervals, ienergy_keV_resolution must be specified.")

        if (self._norm_dimensions == "Em") and self._menergy_keV_resolution is None:
            raise ValueError("Normalization dimensions Em requires menergy_keV_resolution to be specified.")
        
        # Set the arrays

        self._ienergy_keV = ienergy_val
        self._menergy_keV = menergy_arr if menergy_keV is not None else None
        self._scatt_angle_rad = scatt_arr if scatt_angle_rad is not None else None
    
    @staticmethod
    def _subdivide_interval(interval, resolution: float):
        a, b = interval
        diff = b - a

        if diff <= resolution:
            return [a, b]

        num_points = int(np.ceil(diff / resolution)) + 1

        subdivided = np.linspace(a, b, num_points, dtype=float).tolist()

        return subdivided

    def _init_maps(self):
        pol_rad, az_rad = hp.pix2ang(self._map_nside, np.arange(self._map_npix))
        self._coordinates = (pol_rad, az_rad, [], [])
        
        counts = 0
        for ival in self._ienergy_keV:
            if isinstance(ival, float):
                self._coordinates[2].append(ival)
                self._coordinates[3].append(counts)
                counts += 1
            else:
                subdivided = self._subdivide_interval(ival, self._ienergy_keV_resolution)
                self._coordinates[2].append(subdivided)
                length = len(subdivided)
                self._coordinates[3].append(list(range(counts, counts + length)))
                counts += length
    
    def init_setup(self):
        if self._ienergy_keV is None:
            self._set_domain(ienergy_keV=self._ienergy_keV_input, menergy_keV=self._menergy_keV_input, scatt_angle_rad=self._scatt_angle_rad_input)
    
    def init_cache(self):
        if self._maps is None:
            self.init_setup()
            self._init_maps()
            if self._norm_dimensions == "Em":
                self._integration_Em()
            else:
                raise ValueError(f"Unsupported normalization dimensions {self._norm_dimensions}")
    
    def cache_to_file(self, filename: Union[str, Path]):
        self.init_cache()
        
        with h5py.File(str(filename), 'w') as f:
            f.attrs['map_nside'] = self._map_nside
            f.attrs['atol'] = self._atol
            f.attrs['norm_dimensions'] = self._norm_dimensions
            if self._ienergy_keV_resolution is not None:
                f.attrs['ienergy_keV_resolution'] = self._ienergy_keV_resolution
            if self._menergy_keV_resolution is not None:
                f.attrs['menergy_keV_resolution'] = self._menergy_keV_resolution
            if self._range_energy_keV is not None:
                f.attrs['range_energy_keV'] = self._range_energy_keV
            if self._Em_peak_nums is not None:
                f.attrs['Em_peak_nums'] = json.dumps(self._Em_peak_nums)
            if self._Em_peak_widths is not None:
                f.attrs['Em_peak_widths'] = json.dumps(self._Em_peak_widths)
            
            f.attrs['ienergy_keV'] = json.dumps(self._ienergy_keV)
            if self._menergy_keV is not None:
                f.attrs['menergy_keV'] = json.dumps(self._menergy_keV)
            if self._scatt_angle_rad is not None:
                f.attrs['scatt_angle_rad'] = json.dumps(self._scatt_angle_rad)
                
            f.create_dataset('maps', 
                             data=self._maps.numpy(), 
                             compression='gzip', 
                             compression_opts=4)
    
    def cache_from_file(self, filename: Union[str, Path]):
        if not os.path.exists(str(filename)):
            raise FileNotFoundError(f"Cache file {str(filename)} not found.")

        with h5py.File(str(filename), 'r') as f:
            loaded_dim = f.attrs['norm_dimensions']
            if isinstance(loaded_dim, bytes):
                loaded_dim = loaded_dim.decode('utf-8')
                
            if self._norm_dimensions is not None and self._norm_dimensions != loaded_dim:
                 raise ValueError(f"Cache mismatch: File contains {loaded_dim}, "
                                  f"but current model expects {self._norm_dimensions}")
            self._norm_dimensions = loaded_dim
            
            self._map_nside = int(f.attrs['map_nside'])
            self._map_npix = hp.nside2npix(self._map_nside)
            self._atol = float(f.attrs['atol'])
            
            if 'Em_peak_nums' in f.attrs:
                self._Em_peak_nums = tuple(json.loads(f.attrs['Em_peak_nums']))
            else:
                self._Em_peak_nums = None
                
            if 'Em_peak_widths' in f.attrs:
                self._Em_peak_widths = tuple(json.loads(f.attrs['Em_peak_widths']))
            else:
                self._Em_peak_widths = None
                
            if 'ienergy_keV_resolution' in f.attrs:
                self._ienergy_keV_resolution = float(f.attrs['ienergy_keV_resolution'])
            else:
                self._ienergy_keV_resolution = None
                
            if 'menergy_keV_resolution' in f.attrs:
                self._menergy_keV_resolution = float(f.attrs['menergy_keV_resolution'])
            else:
                self._menergy_keV_resolution = None
                
            if 'range_energy_keV' in f.attrs:
                self._range_energy_keV = tuple(f.attrs['range_energy_keV'])
            else:
                self._range_energy_keV = None
            
            self._ienergy_keV = json.loads(f.attrs['ienergy_keV'])
            
            if 'menergy_keV' in f.attrs:
                self._menergy_keV = json.loads(f.attrs['menergy_keV'])
            else:
                self._menergy_keV = None
                
            if 'scatt_angle_rad' in f.attrs:
                self._scatt_angle_rad = json.loads(f.attrs['scatt_angle_rad'])
            else:
                self._scatt_angle_rad = None
            
            self._maps = torch.from_numpy(f['maps'][:])
            self._init_maps()
    
    def _spatial_interpolation(self, q_pol: np.ndarray, q_az: np.ndarray, layer_indices: Union[int, torch.Tensor]) -> torch.Tensor:
        pixels, weights = hp.get_interp_weights(self._map_nside, q_pol, q_az)
        pixels_t = torch.from_numpy(pixels).long()
        weights_t = torch.from_numpy(weights)
        
        y = torch.zeros(len(q_pol), dtype=torch.float64)
        for p in range(4):
            y += self._maps[layer_indices, pixels_t[p]] * weights_t[p]
            
        return y
    
    def _spectral_pchip_interpolation(self, q_e_t: torch.Tensor, e_ref_t: torch.Tensor, 
                                      q_pol: np.ndarray, q_az: np.ndarray, 
                                      map_indices_t: torch.Tensor) -> torch.Tensor:
        M_len = len(e_ref_t)
        
        idx = torch.searchsorted(e_ref_t, q_e_t, right=True) - 1
        idx = torch.clamp(idx, 0, M_len - 2)
        
        I_prev = torch.clamp(idx - 1, min=0)
        I_curr = idx
        I_next = idx + 1
        I_nnext = torch.clamp(idx + 2, max=M_len - 1)
        
        y_prev = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_prev])
        y_curr = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_curr])
        y_next = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_next])
        y_nnext = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_nnext])
        
        hk = torch.diff(e_ref_t)
        
        h_prev = hk[I_prev]
        h_curr = hk[I_curr]
        h_next = hk[torch.clamp(idx + 1, max=M_len - 2)]
        
        m_prev = (y_curr - y_prev) / h_prev
        m_curr = (y_next - y_curr) / h_curr
        m_next = (y_nnext - y_next) / h_next
        
        def standard_pchip(h1, h2, m1, m2):
            condition = (torch.sign(m1) != torch.sign(m2)) | (m1 == 0.0) | (m2 == 0.0)
            w1 = 2.0 * h2 + h1
            w2 = h2 + 2.0 * h1
            
            m1_safe = torch.where(m1 == 0.0, torch.tensor(1.0, dtype=torch.float64), m1)
            m2_safe = torch.where(m2 == 0.0, torch.tensor(1.0, dtype=torch.float64), m2)
            
            whmean = (w1 / m1_safe + w2 / m2_safe) / (w1 + w2)
            return torch.where(condition, torch.tensor(0.0, dtype=torch.float64), 1.0 / whmean)
            
        def edge_case(h0, h1, m0, m1):
            d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            mask_sign = torch.sign(d) != torch.sign(m0)
            mask2 = (torch.sign(m0) != torch.sign(m1)) & (torch.abs(d) > 3.0 * torch.abs(m0))
            mmm = (~mask_sign) & mask2
            d = torch.where(mask_sign, torch.tensor(0.0, dtype=torch.float64), d)
            return torch.where(mmm, 3.0 * m0, d)
            
        d_k = torch.where(
            idx == 0,
            edge_case(h_curr, h_next, m_curr, m_next),
            standard_pchip(h_prev, h_curr, m_prev, m_curr)
        )
        
        d_k1 = torch.where(
            idx == M_len - 2,
            edge_case(h_curr, h_prev, m_curr, m_prev),
            standard_pchip(h_curr, h_next, m_curr, m_next)
        )
        
        t = (q_e_t - e_ref_t[idx]) / h_curr
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        
        return h00 * y_curr + h10 * h_curr * d_k + h01 * y_next + h11 * h_curr * d_k1
    
    def query_normalization(self, pol_rad: ArrayLike, az_rad: ArrayLike, ienergy_keV: ArrayLike) -> np.ndarray:
        self.init_cache()
        
        pol_rad, az_rad, ienergy_keV = [np.atleast_1d(x).ravel() for x in [pol_rad, az_rad, ienergy_keV]]
        
        if not (pol_rad.shape == az_rad.shape == ienergy_keV.shape):
            raise ValueError(
                f"Input shape mismatch: pol_rad {pol_rad.shape}, az_rad {az_rad.shape}, "
                f"and ienergy_keV {ienergy_keV.shape} must all have the same flat length."
            )
        
        results = np.zeros_like(ienergy_keV, dtype=np.float64)
        processed_mask = np.zeros_like(ienergy_keV, dtype=bool)

        energy_blocks = self._coordinates[2]
        index_blocks = self._coordinates[3]

        for block_idx, e_ref in enumerate(energy_blocks):
            map_indices = index_blocks[block_idx]

            if isinstance(e_ref, float):
                mask = np.isclose(ienergy_keV, e_ref, atol=self._atol)
                if np.any(mask):
                    results[mask] = self._spatial_interpolation(pol_rad[mask], az_rad[mask], map_indices).numpy()
                    processed_mask[mask] = True

            else:
                e_ref_arr = np.asarray(e_ref, dtype=np.float64)
                e_min, e_max = e_ref_arr.min(), e_ref_arr.max()

                mask = (ienergy_keV >= e_min - self._atol) & (ienergy_keV <= e_max + self._atol)

                if np.any(mask):
                    q_e_t = torch.from_numpy(np.clip(ienergy_keV[mask], e_min, e_max))
                    e_ref_t = torch.from_numpy(e_ref_arr)
                    map_indices_t = torch.tensor(map_indices, dtype=torch.int64)
                    
                    q_pol = pol_rad[mask]
                    q_az = az_rad[mask]

                    block_results = self._spectral_pchip_interpolation(q_e_t, e_ref_t, q_pol, q_az, map_indices_t)
                    
                    results[mask] = block_results.numpy()
                    processed_mask[mask] = True

        if not np.all(processed_mask):
            unprocessed_energies = ienergy_keV[~processed_mask]
            raise ValueError(
                f"The following energy queries are outside the defined domain/intervals: "
                f"{np.unique(unprocessed_energies)}"
            )

        return results
                    
    def _integration_Em(self):
        context, source, weights, integration_indices, n_spatial, num_ienergy = self._tensor_logic_Em()
        
        active_pool = True
        if isinstance(self._nfdensity, NFNormalizationDensity):
            active_pool = self._nfdensity.active_pool
            if not active_pool:
                self._nfdensity.init_compute_pool()
        
        weighted_density = self._nfdensity.evaluate_density(context, source).to(torch.float64) * weights
        
        if not active_pool:
            self._nfdensity.shutdown_compute_pool()
        
        maps = torch.zeros(n_spatial * num_ienergy, dtype=torch.float64)
        maps.scatter_add_(0, integration_indices, weighted_density)

        self._maps = maps.view(n_spatial, num_ienergy).T
    
    def _tensor_logic_Em(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        pol_rad, az_rad, ienergy_grid, _ = self._coordinates
        ienergy_values = np.hstack(ienergy_grid).astype(np.float32)
        
        nodes_list = []
        weights_list = []
        ie_list = []
        counts_per_ie = []
        
        for ie in ienergy_values:
            n_list = []
            w_list = []
            
            for me in self._menergy_keV:
                n, w = self._nodes_Em(ie, me)
                n_list.append(n)
                w_list.append(w)
            
            nodes_for_ie = torch.cat(n_list).to(torch.float32)
            weights_for_ie = torch.cat(w_list).to(torch.float64)
            
            nodes_list.append(nodes_for_ie)
            weights_list.append(weights_for_ie)
            ie_list.append(torch.full_like(nodes_for_ie, ie))
            counts_per_ie.append(len(nodes_for_ie))
            
        base_me_tensor = torch.cat(nodes_list)
        base_weights_tensor = torch.cat(weights_list)
        base_ie_tensor = torch.cat(ie_list)
        
        pol_tensor = torch.tensor(pol_rad, dtype=torch.float32)
        az_tensor = torch.tensor(az_rad, dtype=torch.float32)
        
        n_spatial = len(pol_tensor)
        n_energy_combos = len(base_me_tensor)
        
        context = torch.stack(
            [
                az_tensor.repeat_interleave(n_energy_combos),
                pol_tensor.repeat_interleave(n_energy_combos),
                base_ie_tensor.repeat(n_spatial),
            ],
            dim=1,
        ).to(torch.float32)
        
        source = base_me_tensor.repeat(n_spatial).to(torch.float32).unsqueeze(1)
        weights = base_weights_tensor.repeat(n_spatial).to(torch.float64)
        
        base_group_indices = torch.repeat_interleave(
            torch.arange(len(ienergy_values)), 
            torch.tensor(counts_per_ie)
        )
        group_offsets = torch.arange(n_spatial) * len(ienergy_values)
        integration_indices = base_group_indices.repeat(n_spatial) + group_offsets.repeat_interleave(n_energy_combos)
        
        return context, source, weights, integration_indices, n_spatial, len(ienergy_values)
    
    def _has_photopeak(self, ienergy_keV: float, menergy_keV: List[float]) -> float:
        energy = ienergy_keV
        contains_peak = (menergy_keV[0] <= energy <= menergy_keV[1])
        return (energy if contains_peak else np.nan)
    
    def _has_escapepeak(self, ienergy_keV: float, menergy_keV: List[float]) -> float:
        energy = ienergy_keV - 511.0
        contains_peak = ((menergy_keV[0] <= energy <= menergy_keV[1]) and (4000.0 >= ienergy_keV >= 1500.0))
        return (energy if contains_peak else np.nan)
    
    def _has_annihilationpeak(self, ienergy_keV: float, menergy_keV: List[float]) -> float:
        energy = 511.0
        contains_peak = ((menergy_keV[0] <= energy <= menergy_keV[1]) and (ienergy_keV >= 2000.0))
        return (energy if contains_peak else np.nan)
    
    @staticmethod
    def _scale_nodes_exp(E1: float, E2: float,
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = E2 - E1

        out_n = (nodes_u + 1).mul(0.5).pow(2).mul(diff).add(E1)
        out_w = (nodes_u + 1).mul(0.5).mul(weights_u).mul(diff)

        return out_n, out_w
    
    @staticmethod
    def _scale_nodes_center(E1: float, E2: float, EC: float,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_left = (nodes_u < 0)
        
        width_left = torch.tensor(EC - E1, dtype=nodes_u.dtype)
        width_right = torch.tensor(E2 - EC, dtype=nodes_u.dtype)

        scale = torch.where(mask_left, width_left, width_right)

        out_n = nodes_u.pow(3).mul(scale).add(EC)
        out_w = nodes_u.pow(2).mul(3).mul(weights_u).mul(scale)

        return out_n, out_w

    @staticmethod
    def _build_nodes(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, w = np.polynomial.legendre.leggauss(degree)
        return torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)
    
    def _Em_peak_params(self, ienergy_keV: float, mode: str) -> Tuple[float, int]:
        if self._Em_peak_nums is None or self._Em_peak_widths is None:
            raise ValueError("The Em normalization requires the peak widths and peak numbers to be set.")
        if mode == 'photo':
            return (np.sqrt(ienergy_keV) * self._Em_peak_widths[0], self._Em_peak_nums[0])
        elif mode == 'annihilation':
            return (np.sqrt(511) * self._Em_peak_widths[1], self._Em_peak_nums[1])
        elif mode == 'escape':
            return (np.sqrt(ienergy_keV-511) * self._Em_peak_widths[2], self._Em_peak_nums[2])
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'photo', 'annihilation', or 'escape'.")

    def _node_logic_Em(self, peaks: List[float], widths: List[Tuple[float, int]], menergy_keV: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        Emin, Emax = menergy_keV
        n_peaks = len(peaks)
        
        def n_fill_peaks(E1, E2):
            diff = E2 - E1
            if diff < self._menergy_keV_resolution:
                return 2
            else:
                return int(np.ceil(diff / self._menergy_keV_resolution))
        
        if n_peaks == 0:
            n_nodes = n_fill_peaks(Emin, Emax)
            n, w = self._build_nodes(n_nodes)
            return self._scale_nodes_exp(Emin, Emax, n, w)
        else:
            nodes = []
            weights = []
            
            diffs = [widths[i][0] for i in range(n_peaks)]
            n_nodes_peaks = [widths[i][1] for i in range(n_peaks)]
            
            El = Emin
            for i in range(n_peaks):
                EC = peaks[i]
                E1 = np.clip(EC - diffs[i], a_min=El, a_max=None)
                Er = Emax if i == n_peaks - 1 else (EC + peaks[i+1])/2
                E2 = np.clip(EC + diffs[i], a_min=None, a_max=Er)
                
                if not np.isclose(El, E1):
                    n_nodes = n_fill_peaks(El, E1)
                    n, w = self._build_nodes(n_nodes)
                    n, w = self._scale_nodes_exp(El, E1, n, w)
                    nodes.append(n)
                    weights.append(w)
                
                n_nodes = n_nodes_peaks[i]
                n, w = self._build_nodes(n_nodes)
                n, w = self._scale_nodes_center(E1, E2, EC, n, w)
                nodes.append(n)
                weights.append(w)
                
                El = E2
            
            if not np.isclose(El, Emax):
                n_nodes = n_fill_peaks(El, Emax)
                n, w = self._build_nodes(n_nodes)
                n, w = self._scale_nodes_exp(El, Emax, n, w)
                nodes.append(n)
                weights.append(w)
            
            return torch.cat(nodes), torch.cat(weights)
            
    def _nodes_Em(self, ienergy_keV: float, menergy_keV: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine which peaks are present
        # Put more peaks with scaled width around them
        # In between peaks are placed with number given my remaining width and node density parameter
        peaks = np.array([self._has_photopeak(ienergy_keV, menergy_keV),
                          self._has_escapepeak(ienergy_keV, menergy_keV),
                          self._has_annihilationpeak(ienergy_keV, menergy_keV)])
        nan_peaks = np.isnan(peaks)
        
        peaks = peaks[~nan_peaks]
        sort_idx_peaks = np.argsort(peaks)
        peaks = list(peaks[sort_idx_peaks])
        peak_types = np.array(['photo', 'escape', 'annihilation'])[~nan_peaks][sort_idx_peaks]
        
        widths = [self._Em_peak_params(ienergy_keV, mode) for mode in peak_types]
        
        Emax = ienergy_keV + self._Em_peak_params(ienergy_keV, "photo")[0]
        
        menergy_keV_local = list(menergy_keV)
        menergy_keV_local[1] = float(np.clip(menergy_keV[1], a_min=None, a_max=Emax))
        
        return self._node_logic_Em(peaks, widths, menergy_keV_local)
