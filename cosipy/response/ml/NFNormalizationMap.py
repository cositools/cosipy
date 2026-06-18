import torch
import numpy as np
import healpy as hp
from typing import Union, Optional, Tuple, List
import json
import os
import h5py
from pathlib import Path

from .NFNormalizationDensity import NFNormalizationDensity
from .NFNormalizationBase import NFNormalizationMapEmBase, MEnergyList, IEnergyList, ArrayLike
from .NFMapInterpolator import SpatialSpectralInterpolator

# TODO: Shift from response only to response + background? -> Would require different integration of background
# TODO: Add option for ARM
# TODO: Add option for more complex functional relationships

class NFNormalizationMap(NFNormalizationMapEmBase[IEnergyList, NFNormalizationDensity]):
    def __init__(self,
                 nfdensity: Optional[NFNormalizationDensity] = None, 
                 ienergy_keV: Optional[IEnergyList] = None,
                 menergy_keV: Optional[MEnergyList] = None,
                 scatt_angle_rad: Optional[MEnergyList] = None,
                 ):
        super().__init__(nfdensity=nfdensity,
                         intvar=ienergy_keV,
                         menergy_keV=menergy_keV,
                         scatt_angle_rad=scatt_angle_rad,
                         )
        
        # Implementation properties
        self._intvar_allow_float: bool = True
        self._intvar_name: str = "ienergy_keV"
        
        # Default parameters
        self._map_nside: int = 32
        self._map_npix: int = hp.nside2npix(self._map_nside)
        self._intvar_resolution: Optional[float] = 1.5
        self._menergy_keV_resolution: Optional[float] = 8.0
        self._range_intvar: Optional[Tuple[float, float]] = (100., 10_000.)
        self._atol: float = 0.1
        self._Em_peak_widths: Optional[Tuple[float, float, float]] = (0.85, 0.70, 0.75)
        self._Em_peak_nums: Optional[Tuple[int, int, int]] = (18, 12, 12)
    
    @property
    def map_nside(self):
        return self._map_nside
    @map_nside.setter
    def map_nside(self, val: int): self.set_integration_parameters(map_nside=val) 
    
    @property
    def range_ienergy_keV(self):
        return self._range_intvar
    @range_ienergy_keV.setter
    def range_ienergy_keV(self, val: Tuple[float, float]): self.set_integration_parameters(range_intvar=val)
    
    @property
    def ienergy_keV_resolution(self):
        return self._intvar_resolution
    @ienergy_keV_resolution.setter
    def ienergy_keV_resolution(self, val: float): self.set_integration_parameters(intvar_resolution=val)
    
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
    
    def set_integration_parameters(self,
                                   map_nside: Optional[int] = -1,
                                   intvar_resolution: Optional[float] = -1.0,
                                   menergy_keV_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[Tuple[float, float]] = None,
                                   atol: Optional[float] = -1.0,
                                   Em_peak_widths: Optional[Tuple[float, float, float]] = None,
                                   Em_peak_nums: Optional[Tuple[int, int, int]] = None):
        
        super().set_integration_parameters(intvar_resolution=intvar_resolution,
                                            menergy_keV_resolution=menergy_keV_resolution,
                                            range_intvar=range_intvar,
                                            atol=atol)
        
        new_map_nside = map_nside if map_nside != -1 else self._map_nside
        new_Em_peak_widths = Em_peak_widths or self._Em_peak_widths
        new_Em_peak_nums = Em_peak_nums or self._Em_peak_nums
        
        if not isinstance(new_map_nside, (int, np.integer)) or (new_map_nside <= 0):
            raise ValueError("map_nside must be a positive integer.")
        
        if not isinstance(new_Em_peak_widths, tuple) or (len(new_Em_peak_widths) != 3) or not isinstance(new_Em_peak_widths[0], (float, int, np.number)) or not isinstance(new_Em_peak_widths[1], (float, int, np.number)) or not isinstance(new_Em_peak_widths[2], (float, int, np.number)):
            raise ValueError("Em_peak_widths must be a tuple of 3 floats.")
        
        if not isinstance(new_Em_peak_nums, tuple) or (len(new_Em_peak_nums) != 3) or not isinstance(new_Em_peak_nums[0], (int, np.integer)) or not isinstance(new_Em_peak_nums[1], (int, np.integer)) or not isinstance(new_Em_peak_nums[2], (int, np.integer)):
            raise ValueError("Em_peak_nums must be a tuple of 3 integers.")
        
        changed = [a != b for a, b in zip(
            [new_map_nside, new_Em_peak_widths, new_Em_peak_nums],
            [self._map_nside, self._Em_peak_widths, self._Em_peak_nums])]
        
        if any(changed):
            if self._nfdensity is None:
                raise RuntimeError(
                    "Cannot change integration parameters that require recalculating the maps "
                    "without providing the 'nfdensity' model."
                )
            self.clear_cache()
        
        self._map_nside = new_map_nside
        self._map_npix = hp.nside2npix(new_map_nside)
        self._Em_peak_nums = new_Em_peak_nums
        self._Em_peak_widths = new_Em_peak_widths

    def _init_maps(self):
        pol_rad, az_rad = hp.pix2ang(self._map_nside, np.arange(self._map_npix))
        self._coordinates = (pol_rad, az_rad, [], [])
        
        counts = 0
        for ival in self._intvar:
            if isinstance(ival, float):
                self._coordinates[2].append(ival)
                self._coordinates[3].append(counts)
                counts += 1
            else:
                subdivided = self._subdivide_interval(ival, self._intvar_resolution)
                self._coordinates[2].append(subdivided)
                length = len(subdivided)
                self._coordinates[3].append(list(range(counts, counts + length)))
                counts += length
    
    def cache_to_file(self, filename: Union[str, Path]):
        self.init_cache()
        
        with h5py.File(str(filename), 'w') as f:
            f.attrs['map_nside'] = self._map_nside
            f.attrs['atol'] = self._atol
            f.attrs['norm_dimensions'] = self._norm_dimensions
            if self._intvar_resolution is not None:
                f.attrs['ienergy_keV_resolution'] = self._intvar_resolution
            if self._menergy_keV_resolution is not None:
                f.attrs['menergy_keV_resolution'] = self._menergy_keV_resolution
            if self._range_intvar is not None:
                f.attrs['range_energy_keV'] = self._range_intvar
            if self._Em_peak_nums is not None:
                f.attrs['Em_peak_nums'] = json.dumps(self._Em_peak_nums)
            if self._Em_peak_widths is not None:
                f.attrs['Em_peak_widths'] = json.dumps(self._Em_peak_widths)
            
            f.attrs['ienergy_keV'] = json.dumps(self._intvar)
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
                self._intvar_resolution = float(f.attrs['ienergy_keV_resolution'])
            else:
                self._intvar_resolution = None
                
            if 'menergy_keV_resolution' in f.attrs:
                self._menergy_keV_resolution = float(f.attrs['menergy_keV_resolution'])
            else:
                self._menergy_keV_resolution = None
                
            if 'range_energy_keV' in f.attrs:
                self._range_intvar = tuple(f.attrs['range_energy_keV'])
            else:
                self._range_intvar = None
            
            self._intvar = json.loads(f.attrs['ienergy_keV'])
            
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
            self._init_interpolator()
    
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
                    results[mask] = self._interpolator._spatial_interpolation(pol_rad[mask], az_rad[mask], map_indices).numpy()
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

                    block_results = self._interpolator._pchip_interpolate(q_e_t, e_ref_t, map_indices_t, q_pol=q_pol, q_az=q_az)
                    
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
        
        weighted_density = self._inference_helper(context, source, weights)
        
        maps = torch.zeros(n_spatial * num_ienergy, dtype=torch.float64)
        maps.scatter_add_(0, integration_indices, weighted_density)

        self._maps = maps.view(n_spatial, num_ienergy).T
        self._init_interpolator()
    
    def _init_interpolator(self):
        self._interpolator = SpatialSpectralInterpolator(self._maps, self._map_nside)
    
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
    
    def _has_photopeak(self, ienergy_keV: float, menergy_keV: List[float]) -> Tuple[float, float, int]:
        width, nodes = self._Em_peak_params(ienergy_keV, 'photo')
        return self._check_peak_active(ienergy_keV, width, nodes, menergy_keV)
    
    def _has_escapepeak(self, ienergy_keV: float, menergy_keV: List[float]) -> Tuple[float, float, int]:
        if not (4000.0 >= ienergy_keV >= 1500.0):
            return np.nan, np.nan, 0
        width, nodes = self._Em_peak_params(ienergy_keV, 'escape')
        return self._check_peak_active(ienergy_keV - 511.0, width, nodes, menergy_keV)
    
    def _has_annihilationpeak(self, ienergy_keV: float, menergy_keV: List[float]) -> Tuple[float, float, int]:
        if not (ienergy_keV >= 2000.0):
            return np.nan, np.nan, 0
        width, nodes = self._Em_peak_params(ienergy_keV, 'annihilation')
        return self._check_peak_active(511.0, width, nodes, menergy_keV)
    
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
            
    def _nodes_Em(self, ienergy_keV: float, menergy_keV: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_peaks = [
            self._has_photopeak(ienergy_keV, menergy_keV),
            self._has_escapepeak(ienergy_keV, menergy_keV),
            self._has_annihilationpeak(ienergy_keV, menergy_keV)
        ]
        active_peaks = [p for p in raw_peaks if p[2] > 0]
        active_peaks.sort(key=lambda x: x[0])
        
        peaks = [p[0] for p in active_peaks]     
        widths = [(p[1], p[2]) for p in active_peaks]
        
        photo_w, _ = self._Em_peak_params(ienergy_keV, "photo")
        Emax_cap = ienergy_keV + photo_w
        menergy_keV_local = list(menergy_keV)
        menergy_keV_local[1] = float(np.clip(menergy_keV[1], a_min=None, a_max=Emax_cap))
        
        return self._node_logic_Em(peaks, widths, menergy_keV_local)
        
        #peaks = peaks[~nan_peaks]
        #sort_idx_peaks = np.argsort(peaks)
        #peaks = list(peaks[sort_idx_peaks])
        #peak_types = np.array(['photo', 'escape', 'annihilation'])[~nan_peaks][sort_idx_peaks]
        #
        #widths = [self._Em_peak_params(ienergy_keV, mode) for mode in peak_types]
        #
        #Emax = ienergy_keV + self._Em_peak_params(ienergy_keV, "photo")[0]
        #
        #menergy_keV_local = list(menergy_keV)
        #menergy_keV_local[1] = float(np.clip(menergy_keV[1], a_min=None, a_max=Emax))
        #
        #return self._node_logic_Em(peaks, widths, menergy_keV_local)
