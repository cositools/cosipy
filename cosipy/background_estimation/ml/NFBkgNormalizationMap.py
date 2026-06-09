from astropy.time import Time
from pathlib import Path
import numpy as np
import torch

from typing import Optional, List, Tuple, Union

from cosipy.response.ml.NFNormalizationBase import NFNormalizationMapEmBase, MEnergyList
from .NFBkgNormalizationDensity import NFBkgNormalizationDensity
from cosipy.response.ml.NFMapInterpolator import TemporalInterpolator

TimeList = List[Tuple[Time, Time]]

class NFBkgNormalizationMap(NFNormalizationMapEmBase[MEnergyList, NFBkgNormalizationDensity]):
    def __init__(self,
                 nfdensity: Optional[NFBkgNormalizationDensity] = None, 
                 time: Optional[TimeList] = None,
                 menergy_keV: Optional[MEnergyList] = None,
                 scatt_angle_rad: Optional[MEnergyList] = None,
                 ):
        super().__init__(nfdensity=nfdensity,
                         intvar=time,
                         menergy_keV=menergy_keV,
                         scatt_angle_rad=scatt_angle_rad,
                         )
        
        # Implementation properties
        self._intvar_allow_float: bool = False
        self._intvar_name: str = "time"
        
        # Default parameters
        self._Em_peak_widths = {
            "w0": 0.30,
            "w1": 0.40
        }
        self._Em_peak_nums = (18, 12)
        self._Em_peaks = [
            # Energy, Width Type, Node Strategy
            (144,   "w0", "pd"),
            (175,   "w0", "p12"),
            (184.5, "w0", "p12"),
            (300,   "w0", "p12"),
            (310,   "w0", "pd"),
            (427,   "w0", "pd"),
            (438,   "w0", "p12"),
            (472,   "w0", "p12"),
            (574,   "w0", "p12"),
            (584,   "w0", "pd"),
            (595.5, "w0", "pd"),
            (752,   "w0", "pd"),
            (810,   "w0", "pd"),
            (843,   "w0", "p12"),
            (872,   "w0", "p12"),
            (987,   "w0", "pd"),
            (1014,  "w0", "p12"),
            (1039,  "w0", "pd"),
            (1076,  "w0", "pd"),
            (1335,  "w0", "pd"),#
            (1367,  "w0", "p18"),#12
            (1432,  "w0", "pd"),
            (1610,  "w0", "pd"),
            (1632,  "w0", "pd"),
            (1763,  "w0", "pd"),
            (1778,  "w0", "pd"),
            (1809,  "w0", "pd"),#
            (1893,  "w0", "pd"),
            (1921,  "w0", "pd"),
            (2208,  "w0", "pd"),
            (2240,  "w0", "pd"),
            (2750,  "w0", "pd"),
            (2822,  "w0", "pd"),
            (198,   "w1", "p18"),
            (397.5, "w1", "p12"),
            (511,   "w1", "p18"),
            (1108,  "w1", "p12")
        ]
        # TODO
        
        self._atol: float = 0.1
        self._intvar_resolution: Optional[float] = 15.0
        self._menergy_keV_resolution: Optional[float] = 8.0
    
    # TODO: getter, setter
    
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
        pass
    
    def cache_from_file(self, filename: Union[str, Path]):
        pass
    
    def query_normalization(self, time: Time) -> np.ndarray:
        self.init_cache()
        
        time = np.atleast_1d(time.utc.unix).ravel()
        
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
        time_values = np.hstack(time_grid).astype(np.float32)
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
        
        time_values_t = torch.from_numpy(time_values).to(torch.float32)
        
        context = torch.repeat_interleave(time_values_t, len_single).to(torch.float32).unsqueeze(1)
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
        if node_type == "p12":
            nodes = self._Em_peak_nums[1]
        elif node_type == "p18":
            nodes = self._Em_peak_nums[0]
        elif node_type == "pd":
            nodes = self._fill_Em_bkg(energy - width, energy + width)
        else:
            raise ValueError(f"Unknown node type strategy key: {node_type}")
        
        return width, nodes
    
    def _nodes_Em(self, menergy_keV: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        active_peaks = []
        
        for energy, w_type, node_type in self._Em_peaks:
            width, nodes = self._Em_peak_params(w_type, node_type, energy)
            p_energy, p_width, p_nodes = self._check_peak_active(energy, width, nodes, menergy_keV)
            if p_nodes > 0:
                active_peaks.append((p_energy, p_width, p_nodes))
        
        active_peaks.sort(key=lambda x: x[0])
        
        peaks = [p[0] for p in active_peaks]     
        widths = [(p[1], p[2]) for p in active_peaks]
        
        menergy_keV_local = list(menergy_keV)
        
        return self._node_logic_Em(peaks, widths, menergy_keV_local)