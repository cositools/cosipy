import json
import os
from pathlib import Path
from typing import Union, Optional, Tuple, Sequence, NamedTuple

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
import torch
from astropy.units import Quantity

from .NFNormalizationDensity import NFNormalizationDensity
from .NFNormalizationBase import NFNormalizationMapEmBase, ArrayLike, NumberTypes, RangeLike
from .NFMapInterpolator import SpatialSpectralInterpolator

class _EnergyBlocks(NamedTuple):
    lo: np.ndarray  
    hi: np.ndarray     
    start: np.ndarray  
    stop: np.ndarray   

class NFNormalizationMap(NFNormalizationMapEmBase[Quantity, NFNormalizationDensity]):
    
    _intvar_allow_float: bool = True
    _intvar_name: str = "ienergy_keV"
    _intvar_unit: u.UnitBase = u.keV
    
    def __init__(self,
                 nfdensity: Optional[NFNormalizationDensity] = None,
                 ienergy: Optional[Quantity] = None,
                 menergy: Optional[Quantity] = None,
                 ):
        super().__init__(nfdensity=nfdensity, intvar=ienergy, menergy=menergy)
        
        # Default parameters
        self._map_nside: int = 32
        self._map_npix: int = hp.nside2npix(self._map_nside)
        self._intvar_resolution: Optional[float] = 1.5
        self._menergy_keV_resolution: Optional[float] = 8.0
        self._atol: float = 0.1
        self._Em_peak_widths: Optional[Tuple[float, float, float]] = (0.85, 0.70, 0.75)
        self._Em_peak_nums: Optional[Tuple[int, int, int]] = (18, 12, 12)
        self._batch_size: int = 262_144
        
        self._range_intvar: Tuple[float, float] = (100., 10_000.)
        self._range_menergy_keV: Tuple[float, float] = (100., 10_000.)
        self._range_scatt_angle_rad: Optional[Tuple[float, float]] = None
    
    @property
    def ienergy(self) -> Optional[Quantity]:
        return self.intvar
    @ienergy.setter
    def ienergy(self, val: Quantity): self.intvar = val
    
    @property
    def map_nside(self) -> int:
        return self._map_nside
    @map_nside.setter
    def map_nside(self, val: int): self.set_integration_parameters(map_nside=val)
    
    @property
    def range_ienergy_keV(self) -> Tuple[float, float]:
        return self._range_intvar
    @range_ienergy_keV.setter
    def range_ienergy_keV(self, val: RangeLike): self.set_integration_parameters(range_intvar=val)
    
    @property
    def range_menergy_keV(self) -> Tuple[float, float]:
        return self._range_menergy_keV
    @range_menergy_keV.setter
    def range_menergy_keV(self, val: RangeLike): self.set_integration_parameters(range_menergy_keV=val)
    
    @property
    def ienergy_keV_resolution(self) -> Optional[float]:
        return self._intvar_resolution
    @ienergy_keV_resolution.setter
    def ienergy_keV_resolution(self, val: float): self.set_integration_parameters(intvar_resolution=val)
    
    @property
    def Em_peak_widths(self) -> Optional[Tuple[float, float, float]]:
        return self._Em_peak_widths
    @Em_peak_widths.setter
    def Em_peak_widths(self, val: Sequence[NumberTypes]): self.set_integration_parameters(Em_peak_widths=val)
    
    @property
    def Em_peak_nums(self) -> Optional[Tuple[int, int, int]]:
        return self._Em_peak_nums
    @Em_peak_nums.setter
    def Em_peak_nums(self, val: Sequence[int]): self.set_integration_parameters(Em_peak_nums=val)
    
    @property
    def batch_size(self) -> int:
        """Maximum number of density evaluations (rows) per call during the integration."""
        return self._batch_size
    @batch_size.setter
    def batch_size(self, val: int): self.set_integration_parameters(batch_size=val)
    
    @staticmethod
    def _parse_triple(name: str, val, integer: bool) -> Tuple:
        kinds = (int, np.integer) if integer else (float, int, np.number)
        if (not isinstance(val, (tuple, list)) or len(val) != 3
                or not all(isinstance(x, kinds) and not isinstance(x, bool) for x in val)
                or not all(x > 0 for x in val)):
            raise ValueError(f"{name} must be a tuple of 3 positive {'integers' if integer else 'floats'}.")
        return tuple(int(x) if integer else float(x) for x in val)
    
    def set_integration_parameters(self,
                                   map_nside: Optional[int] = -1,
                                   intvar_resolution: Optional[float] = -1.0,
                                   menergy_keV_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[RangeLike] = None,
                                   range_menergy_keV: Optional[RangeLike] = None,
                                   atol: Optional[float] = -1.0,
                                   Em_peak_widths: Optional[Sequence[NumberTypes]] = None,
                                   Em_peak_nums: Optional[Sequence[int]] = None,
                                   batch_size: Optional[int] = -1):

        new_map_nside = self._map_nside if map_nside == -1 else map_nside
        new_batch_size = self._batch_size if batch_size == -1 else batch_size
        new_widths = self._Em_peak_widths if Em_peak_widths is None else self._parse_triple("Em_peak_widths", Em_peak_widths, False)
        new_nums = self._Em_peak_nums if Em_peak_nums is None else self._parse_triple("Em_peak_nums", Em_peak_nums, True)
        
        if not isinstance(new_map_nside, (int, np.integer)) or new_map_nside <= 0:
            raise ValueError("map_nside must be a positive integer.")
        if not isinstance(new_batch_size, (int, np.integer)) or new_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        
        changed = (new_map_nside != self._map_nside
                   or new_widths != self._Em_peak_widths
                   or new_nums != self._Em_peak_nums)
        if changed:
            self._require_density_for_change()
        
        super().set_integration_parameters(intvar_resolution=intvar_resolution,
                                           menergy_keV_resolution=menergy_keV_resolution,
                                           range_intvar=range_intvar,
                                           range_menergy_keV=range_menergy_keV,
                                           atol=atol)
        
        self._map_nside = int(new_map_nside)
        self._map_npix = hp.nside2npix(self._map_nside)
        self._Em_peak_widths = new_widths
        self._Em_peak_nums = new_nums
        self._batch_size = int(new_batch_size)
        
        if changed:
            self.clear_cache()

    def _init_maps(self):
        pol_rad, az_rad = hp.pix2ang(self._map_nside, np.arange(self._map_npix))
        
        grids = []
        for (a, b), is_point in zip(self._intvar, self._intvar_is_point):
            if is_point:
                grids.append(np.array([a], dtype=np.float64))
            else:
                grids.append(np.asarray(self._subdivide_interval((a, b), self._intvar_resolution), dtype=np.float64))
        
        lengths = np.array([len(g) for g in grids], dtype=np.int64)
        start = np.cumsum(lengths) - lengths
        order = np.argsort([g[0] for g in grids], kind="stable")
        
        blocks = _EnergyBlocks(lo=np.array([grids[i][0] for i in order]),
                               hi=np.array([grids[i][-1] for i in order]),
                               start=start[order],
                               stop=(start + lengths)[order])
        
        self._coordinates = (pol_rad, az_rad, np.concatenate(grids), blocks)
    
    @staticmethod
    def _intervals_from_json(raw) -> np.ndarray:
        rows = [[float(x), float(x)] if np.isscalar(x) else [float(x[0]), float(x[1])] for x in json.loads(raw)]
        return np.array(rows, dtype=np.float64).reshape(-1, 2)
    
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
            f.attrs['range_ienergy_keV'] = self._range_intvar
            f.attrs['range_menergy_keV'] = self._range_menergy_keV
            if self._Em_peak_nums is not None:
                f.attrs['Em_peak_nums'] = json.dumps(self._Em_peak_nums)
            if self._Em_peak_widths is not None:
                f.attrs['Em_peak_widths'] = json.dumps(self._Em_peak_widths)
            
            f.attrs['ienergy_keV'] = json.dumps(self._intvar.tolist())
            if self._menergy_keV is not None:
                f.attrs['menergy_keV'] = json.dumps(self._menergy_keV.tolist())
            
            f.create_dataset('maps',
                             data=self._maps.numpy(),
                             compression='gzip',
                             compression_opts=4)
    
    def cache_from_file(self, filename: Union[str, Path]):
        if not os.path.exists(str(filename)):
            raise FileNotFoundError(f"Cache file {str(filename)} not found.")

        with h5py.File(str(filename), 'r') as f:
            attrs = f.attrs
            
            loaded_dim = attrs['norm_dimensions']
            if isinstance(loaded_dim, bytes):
                loaded_dim = loaded_dim.decode('utf-8')
            if self._norm_dimensions is not None and self._norm_dimensions != loaded_dim:
                raise ValueError(f"Cache mismatch: File contains {loaded_dim}, "
                                 f"but current model expects {self._norm_dimensions}")
            
            def optional(key, cast):
                return cast(attrs[key]) if key in attrs else None
            
            map_nside = int(attrs['map_nside'])
            atol = float(attrs['atol'])
            peak_nums = optional('Em_peak_nums', lambda x: tuple(json.loads(x)))
            peak_widths = optional('Em_peak_widths', lambda x: tuple(json.loads(x)))
            intvar_resolution = optional('ienergy_keV_resolution', float)
            menergy_resolution = optional('menergy_keV_resolution', float)
            range_intvar = optional('range_ienergy_keV', lambda x: tuple(float(v) for v in x)) or self._range_intvar
            range_menergy = optional('range_menergy_keV', lambda x: tuple(float(v) for v in x)) or self._range_menergy_keV
            intvar = self._intervals_from_json(attrs['ienergy_keV'])
            menergy = optional('menergy_keV', self._intervals_from_json)
            maps = torch.from_numpy(f['maps'][:])
        
        self._norm_dimensions = loaded_dim
        self._map_nside, self._map_npix = map_nside, hp.nside2npix(map_nside)
        self._atol = atol
        self._Em_peak_nums, self._Em_peak_widths = peak_nums, peak_widths
        self._intvar_resolution, self._menergy_keV_resolution = intvar_resolution, menergy_resolution
        self._range_intvar, self._range_menergy_keV = range_intvar, range_menergy
        
        self._intvar, self._menergy_keV, self._scatt_angle_rad = intvar, menergy, None
        self._is_identity_cut = menergy is None or self._is_full_domain(menergy, self._range_menergy_keV)
        
        self._intvar_input = self._export_intervals(intvar, self._intvar_unit)
        self._menergy_input = None if menergy is None else self._export_intervals(menergy, self._menergy_unit)
        self._scatt_angle_input = None
        
        self._init_maps()
        if maps.shape != (len(self._coordinates[2]), self._map_npix):
            self.clear_cache()
            raise ValueError(f"Cache file is inconsistent: maps have shape {tuple(maps.shape)}, expected "
                             f"({len(self._coordinates[2])}, {self._map_npix}).")
        self._maps = maps
        self._init_interpolator()
    
    def query_normalization(self, pol_rad: ArrayLike, az_rad: ArrayLike, ienergy_keV: ArrayLike) -> np.ndarray:
        self.init_setup()
        
        def flat(x, unit):
            if isinstance(x, Quantity):
                x = x.to_value(unit)
            return np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
        
        pol_rad, az_rad, ienergy_keV = flat(pol_rad, u.rad), flat(az_rad, u.rad), flat(ienergy_keV, u.keV)
        
        if not (pol_rad.shape == az_rad.shape == ienergy_keV.shape):
            raise ValueError(
                f"Input shape mismatch: pol_rad {pol_rad.shape}, az_rad {az_rad.shape}, "
                f"and ienergy_keV {ienergy_keV.shape} must all have the same flat length."
            )
        
        if self._is_identity_cut:
            return np.ones_like(ienergy_keV)
        
        self.init_cache()
        
        energies, blocks = self._coordinates[2], self._coordinates[3]
        
        block_id = np.searchsorted(blocks.lo - self._atol, ienergy_keV, side="right") - 1
        inside = (block_id >= 0) & (ienergy_keV <= blocks.hi[np.maximum(block_id, 0)] + self._atol)
        if not inside.all():
            raise ValueError(
                f"The following energy queries are outside the defined domain/intervals: "
                f"{np.unique(ienergy_keV[~inside])}"
            )
        
        results = np.zeros_like(ienergy_keV)
        
        order = np.argsort(block_id, kind="stable")
        ids, first = np.unique(block_id[order], return_index=True)
        last = np.append(first[1:], len(order))
        
        for b, s, e in zip(ids, first, last):
            sel = order[s:e]
            i0, i1 = int(blocks.start[b]), int(blocks.stop[b])
            
            if i1 - i0 == 1:
                results[sel] = self._interpolator._spatial_interpolation(pol_rad[sel], az_rad[sel], i0).numpy()
            else:
                e_ref = energies[i0:i1]
                q_e = torch.from_numpy(np.clip(ienergy_keV[sel], e_ref[0], e_ref[-1]))
                map_indices = torch.arange(i0, i1, dtype=torch.int64)
                results[sel] = self._interpolator._pchip_interpolate(
                    q_e, torch.from_numpy(e_ref), map_indices, q_pol=pol_rad[sel], q_az=az_rad[sel]
                ).numpy()

        return results
    
    def _integration_Em(self):
        pol_rad, az_rad = self._coordinates[0], self._coordinates[1]
        nodes_me, nodes_ie, weights, group, n_ie = self._node_tables_Em()
        
        n_spatial, n_nodes = len(pol_rad), len(nodes_me)
        total_rows = n_spatial * n_nodes
        
        pol_t = torch.as_tensor(pol_rad, dtype=torch.float32)
        az_t = torch.as_tensor(az_rad, dtype=torch.float32)
        
        maps = torch.zeros(n_ie * n_spatial, dtype=torch.float64)
        
        with self._compute_pool(), torch.no_grad():
            for r0 in range(0, total_rows, self._batch_size):
                rows = torch.arange(r0, min(r0 + self._batch_size, total_rows), dtype=torch.int64)
                pix = rows // n_nodes
                node = rows % n_nodes
                
                context = torch.stack([az_t[pix], pol_t[pix], nodes_ie[node]], dim=1)
                source = nodes_me[node].unsqueeze(1)
                weighted_density = self._inference_helper(context, source, weights[node])
                
                maps.scatter_add_(0, group[node] * n_spatial + pix, weighted_density)
        
        self._maps = maps.view(n_ie, n_spatial)
        self._init_interpolator()
    
    def _node_tables_Em(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        energies = self._coordinates[2]
        
        nodes, weights, counts = [], [], []
        for ie in energies:
            parts = [self._nodes_Em(float(ie), me) for me in self._menergy_keV]
            nodes.append(torch.cat([n for n, _ in parts]))
            weights.append(torch.cat([w for _, w in parts]))
            counts.append(len(nodes[-1]))
        
        counts_t = torch.tensor(counts, dtype=torch.int64)
        n_ie = len(energies)
        
        return (torch.cat(nodes).to(torch.float32),
                torch.repeat_interleave(torch.as_tensor(energies, dtype=torch.float32), counts_t),
                torch.cat(weights).to(torch.float64),
                torch.repeat_interleave(torch.arange(n_ie, dtype=torch.int64), counts_t),
                n_ie)
    
    def _init_interpolator(self):
        self._interpolator = SpatialSpectralInterpolator(self._maps, self._map_nside)
    
    def _has_photopeak(self, ienergy_keV: float, menergy_keV: Sequence[float]) -> Tuple[float, float, int]:
        width, nodes = self._Em_peak_params(ienergy_keV, 'photo')
        return self._check_peak_active(ienergy_keV, width, nodes, menergy_keV)
    
    def _has_escapepeak(self, ienergy_keV: float, menergy_keV: Sequence[float]) -> Tuple[float, float, int]:
        if not (4500.0 >= ienergy_keV >= 1500.0):
            return np.nan, np.nan, 0
        width, nodes = self._Em_peak_params(ienergy_keV, 'escape')
        return self._check_peak_active(ienergy_keV - 511.0, width, nodes, menergy_keV)
    
    def _has_annihilationpeak(self, ienergy_keV: float, menergy_keV: Sequence[float]) -> Tuple[float, float, int]:
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
            
    def _nodes_Em(self, ienergy_keV: float, menergy_keV: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_peaks = [
            self._has_photopeak(ienergy_keV, menergy_keV),
            self._has_escapepeak(ienergy_keV, menergy_keV),
            self._has_annihilationpeak(ienergy_keV, menergy_keV)
        ]
        active_peaks = [p for p in raw_peaks if p[2] > 0]
        active_peaks.sort(key=lambda x: x[0])
        
        peaks = [p[0] for p in active_peaks]
        widths = [(p[1], p[2]) for p in active_peaks]
        
        # Above the photopeak the density is negligible: cut the cut there.
        photo_w, _ = self._Em_peak_params(ienergy_keV, "photo")
        Emin, Emax = float(menergy_keV[0]), min(float(menergy_keV[1]), ienergy_keV + photo_w)
        
        if Emax <= Emin:
            return torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
        
        return self._node_logic_Em(peaks, widths, [Emin, Emax])