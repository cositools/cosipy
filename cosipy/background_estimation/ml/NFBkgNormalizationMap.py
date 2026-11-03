import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Mapping, Sequence

import astropy.units as u
import h5py
import numpy as np
import torch
from astropy.time import Time
from astropy.units import Quantity

from cosipy.response.ml.NFNormalizationBase import NFNormalizationMapEmBase, NumberTypes, RangeLike
from .NFBkgNormalizationDensity import NFBkgNormalizationDensity
from cosipy.response.ml.NFMapInterpolator import TemporalInterpolator


@dataclass(frozen=True, order=True)
class BkgPeak:
    energy_keV: float
    width_type: str
    node_type: str
    
_NODE_TYPES = ("base", "medium", "high")

_DEFAULT_PEAKS = (
    (144,   "w0", "base"),
    (175,   "w0", "medium"),
    (184.5, "w0", "medium"),
    (300,   "w0", "medium"),
    (310,   "w0", "base"),
    (427,   "w0", "base"),
    (438,   "w0", "medium"),
    (472,   "w0", "medium"),
    (574,   "w0", "medium"),
    (584,   "w0", "base"),
    (595.5, "w0", "base"),
    (752,   "w0", "base"),
    (810,   "w0", "base"),
    (843,   "w0", "medium"),
    (872,   "w0", "medium"),
    (987,   "w0", "base"),
    (1014,  "w0", "medium"),
    (1039,  "w0", "base"),
    (1076,  "w0", "base"),
    (1335,  "w0", "base"),#
    (1367,  "w0", "medium"),
    (1432,  "w0", "base"),
    (1610,  "w0", "base"),
    (1632,  "w0", "base"),
    (1763,  "w0", "base"),
    (1778,  "w0", "base"),
    (1809,  "w0", "base"),#
    (1893,  "w0", "base"),
    (1921,  "w0", "base"),
    (2208,  "w0", "base"),
    (2240,  "w0", "base"),
    (2750,  "w0", "base"),
    (2822,  "w0", "base"),
    (198,   "w1", "high"),
    (397.5, "w1", "medium"),
    (511,   "w1", "high"),
    (1108,  "w1", "medium")
)


class NFBkgNormalizationMap(NFNormalizationMapEmBase[Time, NFBkgNormalizationDensity]):
    
    _intvar_allow_float: bool = False
    _intvar_name: str = "time"
    _intvar_unit: Optional[u.UnitBase] = None 
    
    def __init__(self,
                 nfdensity: Optional[NFBkgNormalizationDensity] = None,
                 time: Optional[Time] = None,
                 menergy: Optional[Quantity] = None,
                 ):
        super().__init__(nfdensity=nfdensity, intvar=time, menergy=menergy)
        
        # Default parameters
        self._range_intvar: Tuple[float, float] = (1835487300.0, 1843467255.0)  # unix seconds
        self._range_menergy_keV: Tuple[float, float] = (100., 10_000.)
        self._range_scatt_angle_rad: Optional[Tuple[float, float]] = None
        
        self._atol: float = 1e-3
        self._intvar_resolution: Optional[float] = 15.0
        self._menergy_keV_resolution: Optional[float] = 8.0
        
        self._Em_peak_widths: dict = {"w0": 0.30, "w1": 0.40}
        self._Em_peak_nums: Tuple[int, int] = (18, 12)
        self._Em_peaks: Tuple[BkgPeak, ...] = self._parse_peaks(_DEFAULT_PEAKS)
    
    @property
    def time(self) -> Optional[Time]:
        return self.intvar
    @time.setter
    def time(self, val: Time): self.intvar = val
    
    @property
    def range_time(self) -> Tuple[float, float]:
        return self._range_intvar
    @range_time.setter
    def range_time(self, val: RangeLike): self.set_integration_parameters(range_intvar=val)
    
    @property
    def range_menergy_keV(self) -> Tuple[float, float]:
        return self._range_menergy_keV
    @range_menergy_keV.setter
    def range_menergy_keV(self, val: RangeLike): self.set_integration_parameters(range_menergy_keV=val)
    
    @property
    def time_resolution(self) -> Optional[float]:
        """Spacing of the time grid in seconds."""
        return self._intvar_resolution
    @time_resolution.setter
    def time_resolution(self, val: float): self.set_integration_parameters(intvar_resolution=val)
    
    @property
    def Em_peaks(self) -> Tuple[BkgPeak, ...]:
        return self._Em_peaks
    @Em_peaks.setter
    def Em_peaks(self, val: Sequence): self.set_integration_parameters(Em_peaks=val)
    
    @property
    def Em_peak_widths(self) -> dict:
        return dict(self._Em_peak_widths)
    @Em_peak_widths.setter
    def Em_peak_widths(self, val: Mapping[str, NumberTypes]): self.set_integration_parameters(Em_peak_widths=val)
    
    @property
    def Em_peak_nums(self) -> Tuple[int, int]:
        return self._Em_peak_nums
    @Em_peak_nums.setter
    def Em_peak_nums(self, val: Sequence[int]): self.set_integration_parameters(Em_peak_nums=val)
    
    
    @staticmethod
    def _parse_peaks(val: Sequence) -> Tuple[BkgPeak, ...]:
        peaks = []
        for entry in val:
            if isinstance(entry, BkgPeak):
                energy, width_type, node_type = entry.energy_keV, entry.width_type, entry.node_type
            elif isinstance(entry, (tuple, list)) and len(entry) == 3:
                energy, width_type, node_type = entry
            else:
                raise ValueError(f"A peak must be a BkgPeak or (energy_keV, width_type, node_type), got {entry!r}.")
            if isinstance(energy, Quantity):
                energy = energy.to_value(u.keV)
            if isinstance(energy, bool) or not isinstance(energy, (float, int, np.number)):
                raise ValueError(f"Peak energy must be a number in keV, got {energy!r}.")
            peak = BkgPeak(float(energy), width_type, node_type)
            
            if not (np.isfinite(peak.energy_keV) and peak.energy_keV > 0):
                raise ValueError(f"Peak energy must be positive and finite, got {peak.energy_keV}.")
            if not isinstance(peak.width_type, str):
                raise ValueError(f"width_type of the peak at {peak.energy_keV} keV must be a string.")
            if peak.node_type not in _NODE_TYPES:
                raise ValueError(f"Unknown node type {peak.node_type!r} for the peak at {peak.energy_keV} keV, "
                                 f"expected one of {_NODE_TYPES}.")
            peaks.append(peak)
        
        peaks.sort()
        energies = [p.energy_keV for p in peaks]
        if len(set(energies)) != len(energies):
            raise ValueError(f"Peak energies must be unique, duplicates: "
                             f"{sorted({e for e in energies if energies.count(e) > 1})}.")
        return tuple(peaks)
    
    @staticmethod
    def _parse_widths(val: Mapping[str, NumberTypes]) -> dict:
        if (not isinstance(val, Mapping) or len(val) == 0
                or not all(isinstance(k, str) for k in val)
                or not all(isinstance(v, (float, int, np.number)) and not isinstance(v, bool)
                           and np.isfinite(v) and v > 0 for v in val.values())):
            raise ValueError("Em_peak_widths must be a non-empty mapping from width type (str) to a positive float.")
        return {k: float(v) for k, v in val.items()}
    
    @staticmethod
    def _parse_nums(val: Sequence[int]) -> Tuple[int, int]:
        if (not isinstance(val, (tuple, list)) or len(val) != 2
                or not all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 2 for x in val)):
            raise ValueError("Em_peak_nums must be a tuple of 2 integers >= 2 (nodes for 'high' and 'medium').")
        return int(val[0]), int(val[1])
    
    def set_integration_parameters(self,
                                   intvar_resolution: Optional[float] = -1.0,
                                   menergy_keV_resolution: Optional[float] = -1.0,
                                   range_intvar: Optional[RangeLike] = None,
                                   range_menergy_keV: Optional[RangeLike] = None,
                                   atol: Optional[float] = -1.0,
                                   Em_peaks: Optional[Sequence] = None,
                                   Em_peak_widths: Optional[Mapping[str, NumberTypes]] = None,
                                   Em_peak_nums: Optional[Sequence[int]] = None):
        
        new_peaks = self._Em_peaks if Em_peaks is None else self._parse_peaks(Em_peaks)
        new_widths = self._Em_peak_widths if Em_peak_widths is None else self._parse_widths(Em_peak_widths)
        new_nums = self._Em_peak_nums if Em_peak_nums is None else self._parse_nums(Em_peak_nums)
        
        unknown = sorted({p.width_type for p in new_peaks} - set(new_widths))
        if unknown:
            raise ValueError(f"Peaks use width types {unknown} that are not defined in Em_peak_widths "
                             f"({sorted(new_widths)}).")
        
        changed = (new_peaks != self._Em_peaks
                   or new_widths != self._Em_peak_widths
                   or new_nums != self._Em_peak_nums)
        if changed:
            self._require_density_for_change()
        
        super().set_integration_parameters(intvar_resolution=intvar_resolution,
                                            menergy_keV_resolution=menergy_keV_resolution,
                                            range_intvar=range_intvar,
                                            range_menergy_keV=range_menergy_keV,
                                            atol=atol)
        
        self._Em_peaks, self._Em_peak_widths, self._Em_peak_nums = new_peaks, new_widths, new_nums
        
        if changed:
            self.clear_cache()
    
    def _init_maps(self):
        self._coordinates = ([], [])
        
        counts = 0
        for ival in self._intvar:
            subdivided = self._subdivide_interval(ival, self._intvar_resolution)
            self._coordinates[0].append(subdivided)
            length = len(subdivided)
            self._coordinates[1].append(list(range(counts, counts + length)))
            counts += length
    
    def cache_to_file(self, filename: Union[str, Path]):
        self.init_cache()
        
        with h5py.File(str(filename), 'w') as f:
            f.attrs['atol'] = self._atol
            f.attrs['norm_dimensions'] = self._norm_dimensions
            if self._intvar_resolution is not None:
                f.attrs['time_resolution'] = self._intvar_resolution
            if self._menergy_keV_resolution is not None:
                f.attrs['menergy_keV_resolution'] = self._menergy_keV_resolution
            f.attrs['range_time'] = self._range_intvar
            f.attrs['range_menergy_keV'] = self._range_menergy_keV
            
            f.attrs['Em_peaks'] = json.dumps([[p.energy_keV, p.width_type, p.node_type] for p in self._Em_peaks])
            f.attrs['Em_peak_widths'] = json.dumps(self._Em_peak_widths)
            f.attrs['Em_peak_nums'] = json.dumps(self._Em_peak_nums)
            
            f.attrs['time'] = json.dumps(self._intvar.tolist()) 
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
            
            def intervals(raw):
                return np.array(json.loads(raw), dtype=np.float64).reshape(-1, 2)
            
            atol = float(attrs['atol'])
            intvar_resolution = optional('time_resolution', float)
            menergy_resolution = optional('menergy_keV_resolution', float)
            range_intvar = optional('range_time', lambda x: tuple(float(v) for v in x)) or self._range_intvar
            range_menergy = optional('range_menergy_keV', lambda x: tuple(float(v) for v in x)) or self._range_menergy_keV
            peaks = self._parse_peaks(json.loads(attrs['Em_peaks']))
            widths = self._parse_widths(json.loads(attrs['Em_peak_widths']))
            nums = self._parse_nums(json.loads(attrs['Em_peak_nums']))
            intvar = intervals(attrs['time'])
            menergy = optional('menergy_keV', intervals)
            maps = torch.from_numpy(f['maps'][:])
        
        self._norm_dimensions = loaded_dim
        self._atol = atol
        self._intvar_resolution, self._menergy_keV_resolution = intvar_resolution, menergy_resolution
        self._range_intvar, self._range_menergy_keV = range_intvar, range_menergy
        self._Em_peaks, self._Em_peak_widths, self._Em_peak_nums = peaks, widths, nums
        
        self._intvar, self._menergy_keV, self._scatt_angle_rad = intvar, menergy, None
        self._is_identity_cut = menergy is None or self._is_full_domain(menergy, self._range_menergy_keV)
        
        self._intvar_input = self._export_intervals(intvar, self._intvar_unit)
        self._menergy_input = None if menergy is None else self._export_intervals(menergy, self._menergy_unit)
        self._scatt_angle_input = None
        
        try:
            self._init_maps()
            expected = sum(len(grid) for grid in self._coordinates[0])
            if maps.shape != (expected,):
                raise ValueError(f"Cache file is inconsistent: maps have shape {tuple(maps.shape)}, expected ({expected},).")
            self._maps = maps
            self._init_interpolator()
        except Exception:
            self.clear_cache() 
            raise
    
    def query_normalization(self, time: Time) -> np.ndarray:
        self.init_setup()
        
        time = np.atleast_1d(time.utc.unix).ravel()
        
        if self._is_identity_cut:
            return np.ones_like(time, dtype=np.float64)
        
        self.init_cache()
        
        results = np.zeros_like(time, dtype=np.float64)
        processed_mask = np.zeros_like(time, dtype=bool)

        time_blocks = self._coordinates[0]
        index_blocks = self._coordinates[1]
        
        for block_idx, time_ref in enumerate(time_blocks):
            map_indices = index_blocks[block_idx]
            
            time_ref_arr = np.asarray(time_ref, dtype=np.float64)
            time_min, time_max = time_ref_arr.min(), time_ref_arr.max()
            
            mask = (time >= time_min - self._atol) & (time <= time_max + self._atol)
            
            if np.any(mask):
                q_time_t = torch.from_numpy(np.clip(time[mask], time_min, time_max))
                time_ref_t = torch.from_numpy(time_ref_arr)
                map_indices_t = torch.tensor(map_indices, dtype=torch.int64)
                    
                block_results = self._interpolator._pchip_interpolate(q_time_t, time_ref_t, map_indices_t)
                    
                results[mask] = block_results.numpy()
                processed_mask[mask] = True
        
        if not np.all(processed_mask):
            unprocessed_times = time[~processed_mask]
            raise ValueError(
                f"The following time queries (utc.unix) are outside the defined domain/intervals: "
                f"{np.unique(unprocessed_times)}"
            )

        return results
    
    def _integration_Em(self):
        context, source, weights, integration_indices, num_time = self._tensor_logic_Em()
        
        weighted_density = self._inference_helper(context, source, weights)
        self._maps = torch.zeros(num_time, dtype=torch.float64)
        self._maps.scatter_add_(0, integration_indices, weighted_density)
        
        self._init_interpolator()

    def _tensor_logic_Em(self):
        time_grid, _ = self._coordinates
        time_values = np.hstack(time_grid).astype(np.float64) 
        num_time = len(time_values)
        
        n_list = []
        w_list = []
        for me in self._menergy_keV:
            n, w = self._nodes_Em(me)
            n_list.append(n)
            w_list.append(w)
            
        nodes_single = torch.cat(n_list).to(torch.float32)
        weights_single = torch.cat(w_list).to(torch.float64)
        len_single = len(nodes_single)
        
        time_values_t = torch.from_numpy(time_values).to(torch.float64)
        
        context = torch.repeat_interleave(time_values_t, len_single).to(torch.float64).unsqueeze(1)
        source = nodes_single.repeat(num_time).to(torch.float32).unsqueeze(1)
        weights = weights_single.repeat(num_time).to(torch.float64)
        
        integration_indices = torch.repeat_interleave(
            torch.arange(num_time, dtype=torch.int64), 
            len_single
        )
                
        return context, source, weights, integration_indices, len(time_values)

    def _init_interpolator(self):
        self._interpolator = TemporalInterpolator(self._maps)
    
    def _Em_peak_params(self, w_type: str, node_type: str, energy: float) -> Tuple[float, int]:
        width = np.sqrt(energy) * self._Em_peak_widths[w_type]
        if node_type == "medium":
            nodes = self._Em_peak_nums[1]
        elif node_type == "high":
            nodes = self._Em_peak_nums[0]
        elif node_type == "base":
            nodes = self._fill_Em_bkg(energy - width, energy + width)
        else:
            raise ValueError(f"Unknown node type strategy key: {node_type}")
        
        return width, nodes
    
    def _nodes_Em(self, menergy_keV: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        active_peaks = []
        
        for peak in self._Em_peaks:
            width, nodes = self._Em_peak_params(peak.width_type, peak.node_type, peak.energy_keV)
            p_energy, p_width, p_nodes = self._check_peak_active(peak.energy_keV, width, nodes, menergy_keV)
            if p_nodes > 0:
                active_peaks.append((p_energy, p_width, p_nodes))
        
        active_peaks.sort(key=lambda x: x[0])
        
        peaks = [p[0] for p in active_peaks]     
        widths = [(p[1], p[2]) for p in active_peaks]
        
        menergy_keV_local = list(menergy_keV)
        
        return self._node_logic_Em(peaks, widths, menergy_keV_local)