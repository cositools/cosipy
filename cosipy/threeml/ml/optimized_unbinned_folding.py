import copy
import os
import json
from typing import Optional, Iterable, Type, Tuple, List, Union, Dict, Sequence
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sys
from itertools import product, permutations

import numpy as np
import h5py
from astromodels import PointSource
from astropy.coordinates import CartesianRepresentation
from executing import Source
from scoords import SpacecraftFrame

from cosipy import SpacecraftHistory
from cosipy.interfaces.source_response_interface import CachedUnbinnedThreeMLSourceResponseInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces import EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.util.iterables import asarray

from astropy import units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord
from astropy.time import Time

import logging

logger = logging.getLogger(__name__)

import torch
from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.response.ml.NFNormalizationBase import IEnergyList


PeakNodeList = Union[Tuple[int, int], List[Tuple[int, int]]]
DensityNodeList = Union[int, List[int], Tuple[int, ...]]

class UnbinnedThreeMLPointSourceResponseIRFAdaptive(CachedUnbinnedThreeMLSourceResponseInterface):
    
    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf: FarFieldSpectralInstrumentResponseFunctionInterface,
                 sc_history: SpacecraftHistory,
                 show_progress: bool = True,
                 force_energy_node_caching: bool = False,
                 reduce_memory: bool = True):
        
        """
        Will fold the IRF with the point source spectrum by evaluating the IRF at Ei positions adaptively chosen based on characteristic IRF features
        Note that this assumes a smooth flux spectrum
        
        All IRF queries are cached and can be saved to / loaded from a file
        """
        
        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._irf = irf
        self._sc_ori = sc_history
        self.show_progress = show_progress
        self.force_energy_node_caching = force_energy_node_caching
        
        # Default parameters for irf energy node placement
        self._density_integration_nodes = [63,]
        self._total_expectation_resolution = 18.
        self._peak_nodes = [[19, 13],]
        # (photopeak_offset, photopeak_scale, escape_width, missing_energy_scale)
        # half-widths: photopeak = sqrt(Ei + photopeak_offset) * photopeak_scale
        #              escape    = escape_width (constant)
        #              missing_energy (both) = Ei * missing_energy_scale
        self._peak_width_params: Tuple[float, float, float, float, float] = (1200., 0.60, 0.60, 80., 0.15)
        self._center_scale_beta: Tuple[float, float, float] = (2.5, 4.5, 3.0)
        self._TYPE_P, self._TYPE_E, self._TYPE_M, self._TYPE_WM = 0, 1, 2, 3
        self._energy_range = [[100., 10_000.],]
        self._n_intervals = 1
        self._cache_batch_size = 1_000_000
        self._integration_batch_size = 1_000_000
        self._offset: Optional[float] = sys.float_info.min
        
        #TODO: Remove
        self._use_photowidth = True
        self._photofraction = 1.0
        self._epsilonfraction = -0.05
        
        # Per-interval switch: False (default) = adaptive peak+background placement,
        # True = plain linearly-spaced Gauss-Legendre nodes covering the whole interval,
        # ignoring IRF peak locations entirely. Useful for narrow intervals dedicated to
        # resolving a sharp flux feature (e.g. a narrow line) rather than IRF structure.
        self._flat_intervals: List[bool] = [False] * self._n_intervals
        
        # Placeholder for node pool - stored as Tensors
        self._width_tensor: Optional[torch.Tensor] = None
        self._nodes_primary: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_secondary: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        
        self._nodes_bkg_0: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_bkg_1: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_bkg_2: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None
        self._nodes_bkg_3: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None
        
        # Checks to avoid unecessary recomputations
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None
        
        # Cached values
        self._irf_cache: Optional[torch.Tensor] = None # cm^2/rad/sr
        self._irf_energy_node_cache: Optional[np.ndarray] = None # (Optional, only if full batch)
        self._area_cache: Optional[np.ndarray] = None # cm^2*s*keV
        self._area_energy_node_cache: Optional[np.ndarray] = None
        self._exp_events: Optional[float] = None
        self._exp_density: Optional[torch.Tensor] = None
        self._valid_mask_cache: Optional[np.ndarray] = None
        self._valid_events: Optional[int] = None
        self._valid_mask_tensor: Optional[torch.Tensor] = None
        
        # Precomputed spacecraft history - Midpoint
        self._mid_times = self._sc_ori.obstime[:-1] + (self._sc_ori.obstime[1:] - self._sc_ori.obstime[:-1]) / 2
        self._sc_ori_center = self._sc_ori.interp(self._mid_times)
        
        # Precomputed spacecraft history - Simpson
        # t_edges = self._sc_ori.obstime
        # t_mids = t_edges[:-1] + (t_edges[1:] - t_edges[:-1]) / 2
        # all_t, inv_indices = np.unique(Time([t_edges, t_mids]), return_inverse=True)
        # all_t = Time(all_t)
        # self._sc_ori_simpson = self._sc_ori.interp(all_t)
        # edge_indices = inv_indices[:len(t_edges)]
        # mid_indices = inv_indices[len(t_edges):]
        # livetime = self._sc_ori.livetime.to_value(u.s)
        # self._unique_time_weights = np.zeros(len(all_t), dtype=np.float32)
        # np.add.at(self._unique_time_weights, edge_indices[:-1], livetime / 6.0)
        # np.add.at(self._unique_time_weights, edge_indices[1:], livetime / 6.0)
        # np.add.at(self._unique_time_weights, mid_indices, 4.0 * livetime / 6.0)
        
        data_times = self._data.time
        self._n_events = self._data.nevents
        self._unique_unix, self._inv_idx = np.unique(data_times.utc.unix, return_inverse=True)
        unique_times_obj = Time(self._unique_unix, format='unix', scale='utc')
        self._sc_ori_unique = self._sc_ori.interp(unique_times_obj)
        
        interval_ratios = (self._sc_ori.livetime.to_value(u.s) / self._sc_ori.intervals_duration.to_value(u.s))
        bin_indices = np.searchsorted(self._sc_ori.obstime.utc.unix, self._unique_unix, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(self._sc_ori.livetime) - 1)
        unique_ratio = interval_ratios[bin_indices]
        self._livetime_ratio = unique_ratio[self._inv_idx].astype(np.float32)
        
        self._energy_m_keV = torch.as_tensor(asarray(self._data.energy_keV, dtype=np.float32))
        self._phi_rad = torch.as_tensor(asarray(self._data.scattering_angle_rad, dtype=np.float32))
        
        self._lon_scatt = torch.as_tensor(asarray(self._data.scattered_lon_rad_sc, dtype=np.float32))
        self._lat_scatt = torch.as_tensor(asarray(self._data.scattered_lat_rad_sc, dtype=np.float32))
        self._cos_lat_scatt = torch.cos(self._lat_scatt)
        self._sin_lat_scatt = torch.sin(self._lat_scatt)
        self._cos_lon_scatt = torch.cos(self._lon_scatt)
        self._sin_lon_scatt = torch.sin(self._lon_scatt)
        
        # Also runs _check_memory_savings
        self.reduce_memory = reduce_memory
        
        self._max_eps = -1.0
        
    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface
    
    @property
    def force_energy_node_caching(self) -> bool: return self._force_energy_node_caching
    @force_energy_node_caching.setter
    def force_energy_node_caching(self, val):
        if not isinstance(val, bool):
            raise ValueError("force_energy_node_caching must be a boolean")
        self._force_energy_node_caching = val
    
    @property
    def density_integration_nodes(self): return self._density_integration_nodes
    @density_integration_nodes.setter
    def density_integration_nodes(self, val): self.set_integration_parameters(density_integration_nodes=val)

    @property
    def total_expectation_resolution(self) -> float: return self._total_expectation_resolution
    @total_expectation_resolution.setter
    def total_expectation_resolution(self, val): self.set_integration_parameters(total_expectation_resolution=val)

    @property
    def peak_nodes(self): return [tuple(x) for x in self._peak_nodes]
    @peak_nodes.setter
    def peak_nodes(self, val): self.set_integration_parameters(peak_nodes=val)

    @property
    def peak_width_params(self) -> Tuple[float, float, float, float, float]: return self._peak_width_params
    @peak_width_params.setter
    def peak_width_params(self, val): self.set_integration_parameters(peak_width_params=val)

    @property
    def flat_intervals(self) -> List[bool]: return list(self._flat_intervals)
    @flat_intervals.setter
    def flat_intervals(self, val): self.set_flat_intervals(val)

    @property
    def energy_range(self): return [tuple(x) if isinstance(x, list) else x for x in self._energy_range]
    @energy_range.setter
    def energy_range(self, val): self.set_integration_parameters(energy_range=val)

    @property
    def n_intervals(self) -> int: return self._n_intervals

    @property
    def cache_batch_size(self) -> Optional[int]: return self._cache_batch_size
    @cache_batch_size.setter
    def cache_batch_size(self, val): self.set_integration_parameters(cache_batch_size=val)
    
    @property
    def integration_batch_size(self) -> Optional[int]: return self._integration_batch_size
    @integration_batch_size.setter
    def integration_batch_size(self, val): self.set_integration_parameters(integration_batch_size=val)

    @property
    def offset(self) -> Optional[float]: return self._offset
    @offset.setter
    def offset(self, val): self.set_integration_parameters(offset=val)
    
    @property
    def show_progress(self) -> bool: return self._show_progress
    @show_progress.setter
    def show_progress(self, val: bool):
        if not isinstance(val, bool):
            raise ValueError("show_progress must be a boolean")
        self._show_progress = val
    
    def _check_memory_savings(self):
        inefficient = (self._integration_batch_size > (self._n_events * self.total_density_integration_nodes/2))
        if inefficient & self._reduce_memory:
            logger.warning(f"Since integration_batch_size is too large reduce_memory will increase the memory usage! Disable it if this behavior is not desired.")
    @property
    def reduce_memory(self) -> bool: return self._reduce_memory
    @reduce_memory.setter
    def reduce_memory(self, val: bool):
        if not isinstance(val, bool):
            raise ValueError("reduce_memory must be a boolean")
        self._reduce_memory = val
        self._check_memory_savings()
        if not val:
            if self._irf_cache is not None:
                self._irf_cache = torch.as_tensor(self._irf_cache, dtype=torch.float64)
            if self._irf_energy_node_cache is not None:
                self._irf_energy_node_cache = np.asarray(self._irf_energy_node_cache, dtype=np.float64)
        else:
            if self._irf_cache is not None:
                self._irf_cache = torch.as_tensor(self._irf_cache, dtype=torch.float32)
            if self._irf_energy_node_cache is not None:
                self._irf_energy_node_cache = np.asarray(self._irf_energy_node_cache, dtype=np.float32)
    
    @property
    def total_expectation_integration_nodes(self) -> int:
        total = 0
        for x in self._energy_range:
            if isinstance(x, Union[float, int]):
                total += 1
            else:
                total += self._total_expectation_integration_nodes(self._total_expectation_resolution, x[1] - x[0])
        return total
    
    @property
    def total_density_integration_nodes(self) -> int:
        total = 0
        for x in self._energy_range:
            if isinstance(x, float):
                total += 1
        return sum(self._density_integration_nodes) + total
    
    def _total_expectation_integration_nodes(self, resolution: float, diff: float) -> int:
        return int(np.ceil(diff / resolution)) + 1
    
    def _set_integration_ranges(self,
                                density_integration_nodes: Optional[DensityNodeList] = None,
                                peak_nodes: Optional[PeakNodeList] = None,
                                energy_range: Optional[IEnergyList] = None):
        if energy_range is not None:
            if not isinstance(energy_range, list):
                raise ValueError("The energy range must be a list.")
            else:
                energy_range_list = []
                for x in energy_range:
                    if not ((isinstance(x, tuple) and len(x) == 2) or isinstance(x, (float, int, np.number))):
                        raise ValueError("Each element in the energy range must be either a tuple of length 2 or a float.")
                    else:
                        energy_range_list.append([float(x[0]), float(x[1])] if isinstance(x, tuple) else float(x))
                n_intervals = sum(1 for x in energy_range if isinstance(x, tuple) and len(x) == 2)
        else:
            n_intervals = self._n_intervals
            energy_range_list = None
        
        if density_integration_nodes is not None:
            if isinstance(density_integration_nodes, (int, np.integer)):
                density_integration_nodes_list = [int(density_integration_nodes)] * n_intervals
            else:
                if not isinstance(density_integration_nodes, (list, tuple)) or len(density_integration_nodes) != n_intervals or not all(isinstance(n, (int, np.integer)) for n in density_integration_nodes):
                    raise ValueError("If density_integration_nodes is not an integer, it must be a list of integers of the same length as the number of energy intervals.")
                else:
                    density_integration_nodes_list = [int(n) for n in density_integration_nodes]
        else:
            if energy_range is not None and hasattr(self, '_density_integration_nodes'):
                if n_intervals == 0:
                    density_integration_nodes_list = []
                elif len(self._density_integration_nodes) == 1:
                    density_integration_nodes_list = [int(self._density_integration_nodes[0])] * n_intervals
                elif len(self._density_integration_nodes) == n_intervals:
                    density_integration_nodes_list = [int(n) for n in self._density_integration_nodes]
                else:
                    raise ValueError(f"Energy range changed to {n_intervals} intervals, but current density_integration_nodes has length {len(self._density_integration_nodes)}. Please provide a matching list.")
            else:
                density_integration_nodes_list = None
        
        if peak_nodes is not None:
            if isinstance(peak_nodes, tuple) and len(peak_nodes) == 2 and all(isinstance(x, (int, np.integer)) for x in peak_nodes):
                peak_nodes_list = [[int(peak_nodes[0]), int(peak_nodes[1])]] * n_intervals
            else:
                if not all(isinstance(x, tuple) and len(x) == 2 and all(isinstance(y, (int, np.integer)) for y in x) for x in peak_nodes) or (len(peak_nodes) != n_intervals):
                    raise ValueError("Each element in peak_nodes must be a tuple of length 2 containing integers, totalling to the number of energy intervals.")
                peak_nodes_list = [list(int(y) for y in x) for x in peak_nodes]
        else:
            if energy_range is not None and hasattr(self, '_peak_nodes'):
                if n_intervals == 0:
                    peak_nodes_list = []
                elif len(self._peak_nodes) == 1:
                    peak_nodes_list = [list(self._peak_nodes[0])] * n_intervals
                elif len(self._peak_nodes) == n_intervals:
                    peak_nodes_list = [list(x) for x in self._peak_nodes]
                else:
                    raise ValueError(f"Energy range changed to {n_intervals} intervals, but current peak_nodes has length {len(self._peak_nodes)}. Please provide a matching list.")
            else:
                peak_nodes_list = None
            
        return density_integration_nodes_list, peak_nodes_list, energy_range_list, n_intervals

    def set_integration_parameters(self, # TODO: Check that intervals dont overlap
                                   density_integration_nodes: Optional[DensityNodeList] = None,
                                   total_expectation_resolution: float = -1.0,
                                   peak_nodes: Optional[PeakNodeList] = None,
                                   peak_width_params: Optional[Tuple[float, float, float, float, float]] = None,
                                   energy_range: Optional[IEnergyList] = None,
                                   cache_batch_size: Optional[int] = -1,
                                   integration_batch_size: Optional[int] = -1,
                                   offset: float = -1.0):
        
        density_integration_nodes, peak_nodes, energy_range, n_intervals = self._set_integration_ranges(density_integration_nodes, peak_nodes, energy_range)
        
        new_density_integration_nodes = density_integration_nodes if density_integration_nodes is not None else self._density_integration_nodes
        new_total_expectation_resolution = total_expectation_resolution if total_expectation_resolution != -1.0 else self._total_expectation_resolution
        new_peak_nodes = peak_nodes if peak_nodes is not None else self._peak_nodes
        new_peak_width_params = peak_width_params if peak_width_params is not None else self._peak_width_params
        new_range = energy_range if energy_range is not None else self._energy_range
        new_cache_batch = cache_batch_size if cache_batch_size != -1 else self._cache_batch_size
        new_integration_batch = integration_batch_size if integration_batch_size != -1 else self._integration_batch_size
        new_offset = offset if offset != -1.0 else self._offset
        
        irf_affected = (
            new_peak_nodes != self._peak_nodes or 
            new_peak_width_params != self._peak_width_params or 
            new_density_integration_nodes != self._density_integration_nodes or
            new_range != self._energy_range
        )
        
        area_affected = (
            new_total_expectation_resolution != self._total_expectation_resolution or
            new_range != self._energy_range
        )
        
        if irf_affected:
            self._irf_cache = self._irf_energy_node_cache = self._width_tensor = None
            self._nodes_primary = self._nodes_secondary = None
            self._nodes_bkg_0 = self._nodes_bkg_1 = self._nodes_bkg_2 = self._nodes_bkg_3 = None
        
        if area_affected:
            self._area_cache = self._area_energy_node_cache = None
        
        if len(new_peak_width_params) != 5:
            raise ValueError("peak_width_params must have exactly 5 entries: "
                              "(photopeak_offset, 2 x photopeak_scale, escape_width, missing_energy_scale).")
        if any(v < 0 for v in new_peak_width_params):
            raise ValueError("peak_width_params entries must be non-negative.")
        
        if n_intervals > 0:
            if any(new_density_integration_nodes[i] < (new_peak_nodes[i][0] + 2 * new_peak_nodes[i][1] + 3) for i in range(n_intervals)):
                raise ValueError("Too many nodes per peak compared to the total number or peaks!")

            if any(n < 1 for n in new_density_integration_nodes) or any(n < 1 for n in np.hstack(new_peak_nodes)):
                raise ValueError("The number of energy nodes must be at least 1.")
        
        energy_intervals = [i for i in new_range if isinstance(i, list)]
        
        if n_intervals > 0:
            smallest_interval = min((energy_intervals[i][1] - energy_intervals[i][0]) for i in range(n_intervals))
            if (new_total_expectation_resolution > smallest_interval) or (new_total_expectation_resolution <= 0):
                raise ValueError("The total expectation resolution must be positive and smaller than the energy range.")
            if any(i[1] <= i[0] for i in energy_intervals):
                raise ValueError("The initial energy interval needs to be increasing!")
        else:
            if new_total_expectation_resolution <= 0:
                raise ValueError("The total expectation resolution must be positive.")
        
        new_total_expectation_integration_nodes = sum(self._total_expectation_integration_nodes(new_total_expectation_resolution, i[1] - i[0]) for i in energy_intervals)
        new_max_nodes = max(new_total_expectation_integration_nodes, sum(new_density_integration_nodes)) + (len(new_range) - n_intervals)
        
        if (new_cache_batch is not None) and (new_cache_batch < new_max_nodes):
            raise ValueError(f"The cache batch size cannot be smaller than the number of integration nodes ({new_max_nodes}).")
        
        if (new_integration_batch is not None) and (new_integration_batch < new_max_nodes):
            raise ValueError(f"The integration batch size cannot be smaller than the number of integration nodes ({new_max_nodes}).")
        
        if (new_offset is not None) and (new_offset < 0):
            raise ValueError("The offset cannot be negative.")
            
        self._density_integration_nodes = new_density_integration_nodes
        self._total_expectation_resolution = new_total_expectation_resolution
        self._peak_nodes = new_peak_nodes
        self._peak_width_params = new_peak_width_params
        self._energy_range = new_range
        self._cache_batch_size = new_cache_batch if new_cache_batch is not None else (self._n_events * new_max_nodes)
        self._integration_batch_size = new_integration_batch if new_integration_batch is not None else (self._n_events * new_max_nodes)
        self._offset = new_offset
        self._n_intervals = n_intervals
        self._check_memory_savings()
    
    def set_flat_intervals(self, flat_intervals: Union[bool, List[bool]]):
        """
        Mark specific energy intervals to be integrated with a plain, linearly-spaced set
        of Gauss-Legendre nodes covering the whole interval, instead of the adaptive
        peak+background placement. Since these nodes don't depend on any per-event peak
        location, they are identical for every event and are broadcast rather than computed
        per event, so flat intervals are also cheaper than adaptive ones.
 
        Intended for narrow intervals dedicated to resolving a sharp flux feature (e.g. a
        narrow line), where the IRF-peak-based placement concentrates nodes on IRF response
        structure rather than on the flux's own narrow structure.
        """
        n_intervals = sum(1 for x in self._energy_range if isinstance(x, list))
        
        if isinstance(flat_intervals, bool):
            new_flat_intervals = [flat_intervals] * n_intervals
        else:
            if (not isinstance(flat_intervals, (list, tuple))
                or len(flat_intervals) != n_intervals
                or not all(isinstance(b, bool) for b in flat_intervals)):
                raise ValueError("flat_intervals must be a bool or a list of bools matching the number of energy intervals.")
            new_flat_intervals = list(flat_intervals)
        
        self._flat_intervals = new_flat_intervals
        # Forces _init_node_pool() to run again the next time the density is requested -
        # same mechanism `irf_affected` uses in set_integration_parameters(). Not strictly
        # necessary here since flat mode reuses the existing _nodes_bkg_0 pool untouched,
        # but kept for consistency / to be robust to future changes.
        self._irf_cache = self._irf_energy_node_cache = None
    
    @staticmethod
    def _build_nodes(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, w = np.polynomial.legendre.leggauss(degree)
        return torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), torch.as_tensor(w, dtype=torch.float32).unsqueeze(0)
    
    def _build_split_nodes(self, remaining: int, groups: int):
        q, r = divmod(remaining, groups)
        return [self._build_nodes(q + (1 if i < r else 0)) for i in range(groups)]
    
    def _init_node_pool(self):
        self._width_tensor = []
        self._nodes_primary = []
        self._nodes_secondary = []
        self._nodes_bkg_0 = []
        self._nodes_bkg_1 = []
        self._nodes_bkg_2 = []
        self._nodes_bkg_3 = []
        
        n_intervals = len(self._density_integration_nodes)
        if len(self._flat_intervals) != n_intervals:
            # Re-sync after energy_range/n_intervals changed via set_integration_parameters()
            self._flat_intervals = [False] * n_intervals
        
        for k in range(len(self._density_integration_nodes)):
            n_density = self._density_integration_nodes[k]
            p_nodes = self._peak_nodes[k]
            
            w_tensor = torch.tensor(self._peak_width_params, dtype=torch.float32)
            self._width_tensor.append(w_tensor)

            self._nodes_primary.append(self._build_nodes(p_nodes[0]))
            self._nodes_secondary.append(self._build_nodes(p_nodes[1]))
            
            # Also reused as the flat-mode node pool: identical plain n_density-node
            # Gauss-Legendre set, just mapped linearly instead of exponentially in _get_nodes.
            self._nodes_bkg_0.append(self._build_nodes(n_density))

            self._nodes_bkg_1.append(self._build_nodes(n_density - p_nodes[0]))

            self._nodes_bkg_2.append((
                self._build_split_nodes(n_density - p_nodes[0] - p_nodes[1], 2), # has_photopeak
                self._build_split_nodes(n_density - p_nodes[1], 2),             # ~has_photopeak
                ))

            self._nodes_bkg_3.append((
                self._build_split_nodes(n_density - p_nodes[0] - 2 * p_nodes[1], 3), # has_photopeak
                self._build_split_nodes(n_density - 2 * p_nodes[1], 3),             # ~has_photopeak
                ))

    @staticmethod
    def _scale_nodes_exp(E1: torch.Tensor, E2: torch.Tensor, 
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = E2 - E1

        out_n = (nodes_u + 1).mul(0.5).pow(2).mul(diff).add(E1)
        out_w = (nodes_u + 1).mul(0.5).mul(weights_u).mul(diff)

        return out_n, out_w
    
    @staticmethod
    def _scale_nodes_linear(E1: torch.Tensor, E2: torch.Tensor,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plain affine Gauss-Legendre mapping from [-1, 1] to [E1, E2], with no bias
        toward either edge and no dependence on any peak location."""
        diff = E2 - E1
 
        out_n = (nodes_u + 1).mul(0.5).mul(diff).add(E1)
        out_w = weights_u.mul(0.5).mul(diff)
 
        return out_n, out_w
    
    @staticmethod
    def _scale_nodes_center(E1: torch.Tensor, E2: torch.Tensor, EC: torch.Tensor,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor,
                            beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        width_left = (EC - E1)
        width_right = (E2 - EC)
        total_width = width_left + width_right

        is_zero = (total_width < 1e-9)
        safe_total = torch.where(is_zero, torch.ones_like(total_width), total_width)

        f = width_left / safe_total
        u0 = 2.0 * f - 1.0
        u0 = torch.clamp(u0, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        mask_left = (nodes_u < u0)
        t_left  = (nodes_u - u0) / (u0 + 1.0)
        t_right = (nodes_u - u0) / (1.0 - u0)
        t = torch.where(mask_left, t_left, t_right)

        scale = torch.where(mask_left, width_left, width_right)
        jac_denom = torch.where(mask_left, (u0 + 1.0), (1.0 - u0))

        sinh_beta = torch.sinh(beta)
        out_n = EC + scale * torch.sinh(beta * t) / sinh_beta
        out_w = weights_u * scale * beta * torch.cosh(beta * t) / sinh_beta / jac_denom

        w_sum = out_w.sum(dim=-1, keepdim=True)
        sum_is_zero = (w_sum.abs() < 1e-12)
        safe_w_sum = torch.where(sum_is_zero, torch.ones_like(w_sum), w_sum)
        out_w = out_w * total_width / safe_w_sum

        out_n = torch.where(is_zero | sum_is_zero, EC, out_n)
        out_w = torch.where(is_zero | sum_is_zero, torch.zeros_like(out_w), out_w)

        return out_n, out_w

    @staticmethod
    def _scale_nodes_log(E1: torch.Tensor, E2: torch.Tensor,
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_E1 = torch.log10(E1)
        log_E2 = torch.log10(E2)
        scale = 0.5 * (log_E2 - log_E1)

        out_n = torch.pow(10, nodes_u.mul(scale).add(0.5 * (log_E1 + log_E2)))
        out_w = out_n.mul(weights_u).mul(scale).mul(torch.log(torch.tensor(10.0)))

        return out_n, out_w
    
    @staticmethod
    def _get_escape_peak(energy_m_keV: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        E2 = 511.0 / (1.0 + 511.0 / energy_m_keV - torch.cos(phi_rad))
        energy = energy_m_keV + 1022.0 - E2
        
        accept = (energy > 1600.0) & (energy_m_keV < energy)
        return torch.where(accept, energy, torch.tensor(float('nan'), dtype=torch.float32))
    
    def _get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad,
                                 photopeak_offset: float, photopeak_scale_right: float,
                                 use_photowidth: bool, photofraction: float, epsilonfraction: float,
                                 inverse: bool = False) -> torch.Tensor:
        cos_geo = torch.cos(phi_geo_rad)
        cos_phi = torch.cos(phi_rad)
        
        if inverse:
            denom = 2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            root = torch.sqrt(energy_m_keV * (cos_geo - 1) * (-2044.0 - 5 * energy_m_keV + energy_m_keV * cos_geo + 4 * energy_m_keV * cos_phi))
            energy = 511.0 * (energy_m_keV - energy_m_keV * cos_geo + root) / denom
        else:
            denom = 2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            root = torch.sqrt(energy_m_keV**2 * (cos_geo - 1) * (cos_phi - 1) *
                              ((1022.0 + energy_m_keV)**2 - energy_m_keV * (2044.0 + energy_m_keV) * cos_phi - 2 * energy_m_keV**2 * cos_geo * torch.sin(phi_rad/2)**2))
            energy = (energy_m_keV**2 * (1 - cos_geo - cos_phi + cos_phi * cos_geo) + root) / denom

        if use_photowidth:
            hw_P_right = torch.sqrt(energy_m_keV.clamp(min=-photopeak_offset + 1e-6) + photopeak_offset) * photopeak_scale_right
            accept = energy >= (energy_m_keV + hw_P_right * photofraction)
        else:
            accept = energy_m_keV/energy - 1 < epsilonfraction
        return torch.where(accept, energy, torch.tensor(float('nan'), dtype=torch.float32))
    
    def _peak_half_width(self, peak_type: int, peak_val: torch.Tensor, width_params) -> Tuple[torch.Tensor, torch.Tensor]:
        photopeak_offset, photopeak_scale_left, photopeak_scale_right, escape_width, missing_energy_scale = width_params
        if peak_type == self._TYPE_P:
            base = torch.sqrt(peak_val.clamp(min=-photopeak_offset + 1e-6) + photopeak_offset)
            return base * photopeak_scale_left, base * photopeak_scale_right
        elif peak_type == self._TYPE_E:
            w = torch.full_like(peak_val, escape_width)
            return w, w
        else:
            w = peak_val * missing_energy_scale
            return w, w
    
    def init_cache(self):
        self._update_cache()
    
    def clear_cache(self):
        self._irf_cache = None
        self._irf_energy_node_cache = None
        self._area_cache = None
        self._area_energy_node_cache = None
        self._exp_events = None
        self._exp_density = None
        self._valid_mask_cache = None
        self._valid_events = None
        self._valid_mask_tensor = None
        
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None
    
    def set_source(self, source: Source):
        if not isinstance(source, PointSource):
            raise TypeError("Please provide a PointSource!")

        self._source = source
    
    def copy(self) -> CachedUnbinnedThreeMLSourceResponseInterface:
        new_instance = copy.copy(self)
        new_instance.clear_cache()
        new_instance._source = None
        
        return new_instance
    
    @staticmethod
    def _earth_occ(source_coord: SkyCoord, ori: SpacecraftHistory) -> np.ndarray:
        dist_earth_center = ori.location.spherical.distance.km
        max_angle = np.pi - np.arcsin(c.R_earth.to(u.km).value/dist_earth_center)
        src_angle = source_coord.separation(ori.earth_zenith)
        return (src_angle.to(u.rad).value < max_angle).astype(np.float32)
    
    @staticmethod
    def _get_target_in_sc_frame(source_coord: SkyCoord, ori: SpacecraftHistory) -> SkyCoord:
        src_in_sc_frame = SkyCoord(np.dot(ori.attitude.rot.inv().as_matrix(), source_coord.transform_to(ori.attitude.frame).cartesian.xyz.value),
                                   representation_type = 'cartesian', frame = SpacecraftFrame())

        src_in_sc_frame.representation_type = 'spherical'
        return src_in_sc_frame
    
    def _compute_area(self):
        coord = self._source.position.sky_coord
        n_energy = self.total_expectation_integration_nodes
        
        e_n, e_w = [], []

        for x in self._energy_range:
            if isinstance(x, (float, int)):
                e_n.append([float(x)])
                e_w.append([1.0])
            else:
                E1, E2 = torch.tensor(x[0]), torch.tensor(x[1])
                n_nodes = self._total_expectation_integration_nodes(self._total_expectation_resolution, E2 - E1)

                n_np, w_np = np.polynomial.legendre.leggauss(n_nodes)

                n_torch = torch.from_numpy(n_np)
                w_torch = torch.from_numpy(w_np)

                n, w = self._scale_nodes_log(E1, E2, n_torch, w_torch)

                e_n.append(n.numpy())
                e_w.append(w.numpy())

        self._area_energy_node_cache = np.concatenate(e_n).astype(np.float64)
        e_n = self._area_energy_node_cache.astype(np.float32)
        e_w = np.concatenate(e_w).astype(np.float32)

        # Midpoint
        sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_center)
        earth_occ_index = self._earth_occ(coord, self._sc_ori_center)
        
        combined_time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(np.float32) * earth_occ_index
        
        # Simpson
        # sc_coord_sph = self._sc_ori_simpson.get_target_in_sc_frame(coord)
        # earth_occ_index = self._earth_occ(coord, self._sc_ori_simpson)
        
        # combined_time_weights = (self._unique_time_weights * earth_occ_index).astype(np.float32)

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

        n_time = len(lon_ph_rad)
        batch_size_time = self._cache_batch_size // n_energy

        total_area = np.zeros(n_energy, dtype=np.float64)
        
        max_batch_total = n_energy * min(batch_size_time, n_time)
        batch_lons_buffer = np.empty(max_batch_total, dtype=np.float32)
        batch_lats_buffer = np.empty(max_batch_total, dtype=np.float32)
        batch_energies_buffer = np.empty(max_batch_total, dtype=np.float32)
        
        for i in tqdm(range(0, n_time, batch_size_time), 
                      disable=(not self.show_progress),
                      desc="Caching the effective area", 
                      smoothing=0.2, 
                      leave=False):
            start = i
            end = min(i + batch_size_time, n_time)
            current_n_time = end - start
            current_total = current_n_time * n_energy

            batch_lons_buffer[:current_total].reshape(current_n_time, n_energy)[:] = lon_ph_rad[start:end, np.newaxis]
            batch_lats_buffer[:current_total].reshape(current_n_time, n_energy)[:] = lat_ph_rad[start:end, np.newaxis]
            batch_energies_buffer[:current_total].reshape(current_n_time, n_energy)[:] = e_n
            
            photons = PhotonListWithDirectionAndEnergyInSCFrame(
                batch_lons_buffer[:current_total],
                batch_lats_buffer[:current_total],
                batch_energies_buffer[:current_total]
                )
            
            eff_areas_flat = asarray(self._irf.effective_area_cm2(photons), dtype=np.float32)
            eff_areas_grid = eff_areas_flat.reshape(current_n_time, n_energy)

            total_area += np.einsum('ij,i,j->j', 
                                    eff_areas_grid, 
                                    combined_time_weights[start:end], 
                                    e_w.ravel())

        self._area_cache = total_area
    
    def _peak_scale_args(self, type_col: torch.Tensor):
        beta = torch.where(type_col == self._TYPE_P,
                            torch.full_like(type_col, self._center_scale_beta[0], dtype=torch.float32),
                            torch.where(type_col == self._TYPE_E,
                                        torch.full_like(type_col, self._center_scale_beta[1], dtype=torch.float32),
                                        torch.full_like(type_col, self._center_scale_beta[2], dtype=torch.float32)))
        return beta

    def _fill_nodes(self, nodes_out, weights_out, indices, mode, has_photopeak,
                    sorted_peaks, delta_left, delta_right, sorted_type,
                    intervals_idx, current_offset, Emin, Emax):

        Emin_t = torch.full((len(indices), 1), Emin, dtype=torch.float32)
        Emax_t = torch.full((len(indices), 1), Emax, dtype=torch.float32)

        if mode == 0:
            c = 0
            w = self._nodes_bkg_0[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(Emin_t, Emax_t, *self._nodes_bkg_0[intervals_idx])
            nodes_out[indices, current_offset + c : current_offset + c + w] = n_res
            weights_out[indices, current_offset + c : current_offset + c + w] = w_res

        elif mode == 1:
            E1 = (sorted_peaks[:, 0] - delta_left[:, 0]).clamp(min=Emin)
            E2 = (sorted_peaks[:, 0] + delta_right[:, 0]).clamp(max=Emax)
            EC = sorted_peaks[:, 0]
            E1, E2, EC = [E.view(-1, 1) for E in (E1, E2, EC)]
            beta = self._peak_scale_args(sorted_type[:, 0:1])

            if torch.any(has_photopeak):
                c = 0
                w = self._nodes_primary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(E1[has_photopeak], E2[has_photopeak], EC[has_photopeak],
                                                          *self._nodes_primary[intervals_idx],
                                                          beta[has_photopeak])
                nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_bkg_1[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(E2[has_photopeak], Emax_t[has_photopeak], *self._nodes_bkg_1[intervals_idx])
                nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            if torch.any(~has_photopeak):
                c = 0
                w = self._nodes_bkg_2[intervals_idx][1][0][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(Emin_t[~has_photopeak], E1[~has_photopeak], *self._nodes_bkg_2[intervals_idx][1][0])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(E1[~has_photopeak], E2[~has_photopeak], EC[~has_photopeak],
                                                          *self._nodes_secondary[intervals_idx],
                                                          beta[~has_photopeak])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_bkg_2[intervals_idx][1][1][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(E2[~has_photopeak], Emax_t[~has_photopeak], *self._nodes_bkg_2[intervals_idx][1][1])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

        elif mode == 2:
            center_peak = (sorted_peaks[:, 0] + sorted_peaks[:, 1]) / 2
            E1 = (sorted_peaks[:, 0] - delta_left[:, 0]).clamp(min=Emin)
            E3 = (sorted_peaks[:, 1] - delta_left[:, 1]).clamp(min=center_peak)
            E2 = (sorted_peaks[:, 0] + delta_right[:, 0]).clamp(max=E3)
            E4 = (sorted_peaks[:, 1] + delta_right[:, 1]).clamp(max=Emax)
            EC1 = sorted_peaks[:, 0]; EC2 = sorted_peaks[:, 1]
            E1, E2, E3, E4, EC1, EC2 = [E.view(-1, 1) for E in (E1, E2, E3, E4, EC1, EC2)]
            beta1 = self._peak_scale_args(sorted_type[:, 0:1])
            beta2 = self._peak_scale_args(sorted_type[:, 1:2])

            if torch.any(has_photopeak):
                c = 0
                w = self._nodes_primary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(E1[has_photopeak], E2[has_photopeak], EC1[has_photopeak],
                                                          *self._nodes_primary[intervals_idx], beta1[has_photopeak])
                nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_bkg_2[intervals_idx][0][0][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(E2[has_photopeak], E3[has_photopeak], *self._nodes_bkg_2[intervals_idx][0][0])
                nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(E3[has_photopeak], E4[has_photopeak], EC2[has_photopeak],
                                                          *self._nodes_secondary[intervals_idx], beta2[has_photopeak])
                nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_bkg_2[intervals_idx][0][1][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(E4[has_photopeak], Emax_t[has_photopeak], *self._nodes_bkg_2[intervals_idx][0][1])
                nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            if torch.any(~has_photopeak):
                c = 0
                w = self._nodes_bkg_3[intervals_idx][1][0][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(Emin_t[~has_photopeak], E1[~has_photopeak], *self._nodes_bkg_3[intervals_idx][1][0])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(E1[~has_photopeak], E2[~has_photopeak], EC1[~has_photopeak],
                                                          *self._nodes_secondary[intervals_idx], beta1[~has_photopeak])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_bkg_3[intervals_idx][1][1][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(E2[~has_photopeak], E3[~has_photopeak], *self._nodes_bkg_3[intervals_idx][1][1])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(E3[~has_photopeak], E4[~has_photopeak], EC2[~has_photopeak],
                                                          *self._nodes_secondary[intervals_idx], beta2[~has_photopeak])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

                c += w
                w = self._nodes_bkg_3[intervals_idx][1][2][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(E4[~has_photopeak], Emax_t[~has_photopeak], *self._nodes_bkg_3[intervals_idx][1][2])
                nodes_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = n_res
                weights_out[indices[~has_photopeak], current_offset + c : current_offset + c + w] = w_res

        elif mode == 3:
            center_peak_1 = (sorted_peaks[:, 0] + sorted_peaks[:, 1]) / 2
            center_peak_2 = (sorted_peaks[:, 1] + sorted_peaks[:, 2]) / 2
            E1 = (sorted_peaks[:, 0] - delta_left[:, 0]).clamp(min=Emin)
            E3 = (sorted_peaks[:, 1] - delta_left[:, 1]).clamp(min=center_peak_1)
            E2 = (sorted_peaks[:, 0] + delta_right[:, 0]).clamp(max=E3)
            E4 = (sorted_peaks[:, 1] + delta_right[:, 1]).clamp(max=center_peak_2)
            E5 = (sorted_peaks[:, 2] - delta_left[:, 2]).clamp(min=E4)
            E6 = (sorted_peaks[:, 2] + delta_right[:, 2]).clamp(max=Emax)
            EC1, EC2, EC3 = [sorted_peaks[:, i] for i in range(3)]
            E1, E2, E3, E4, E5, E6, EC1, EC2, EC3 = [E.view(-1, 1) for E in (E1, E2, E3, E4, E5, E6, EC1, EC2, EC3)]
            beta1 = self._peak_scale_args(sorted_type[:, 0:1])
            beta2 = self._peak_scale_args(sorted_type[:, 1:2])
            beta3 = self._peak_scale_args(sorted_type[:, 2:3])

            c = 0
            w = self._nodes_primary[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_center(E1[has_photopeak], E2[has_photopeak], EC1[has_photopeak],
                                                      *self._nodes_primary[intervals_idx], beta1[has_photopeak])
            nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
            weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            c += w
            w = self._nodes_bkg_3[intervals_idx][0][0][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E2[has_photopeak], E3[has_photopeak], *self._nodes_bkg_3[intervals_idx][0][0])
            nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
            weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            c += w
            w = self._nodes_secondary[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_center(E3[has_photopeak], E4[has_photopeak], EC2[has_photopeak],
                                                      *self._nodes_secondary[intervals_idx], beta2[has_photopeak])
            nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
            weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            c += w
            w = self._nodes_bkg_3[intervals_idx][0][1][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E4[has_photopeak], E5[has_photopeak], *self._nodes_bkg_3[intervals_idx][0][1])
            nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
            weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            c += w
            w = self._nodes_secondary[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_center(E5[has_photopeak], E6[has_photopeak], EC3[has_photopeak],
                                                      *self._nodes_secondary[intervals_idx], beta3[has_photopeak])
            nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
            weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res

            c += w
            w = self._nodes_bkg_3[intervals_idx][0][2][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E6[has_photopeak], Emax_t[has_photopeak], *self._nodes_bkg_3[intervals_idx][0][2])
            nodes_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = n_res
            weights_out[indices[has_photopeak], current_offset + c : current_offset + c + w] = w_res
        else:
            raise ValueError(f"Unknown folding mode {mode}")
    
    def _get_nodes(self, energy_m_keV, phi_rad, phi_geo_rad, phi_igeo_rad):
        energy_m_keV = energy_m_keV.view(-1, 1)
        phi_rad = phi_rad.view(-1, 1)
        phi_geo_rad = phi_geo_rad.view(-1, 1)
        phi_igeo_rad = phi_igeo_rad.view(-1, 1)

        batch_size = energy_m_keV.shape[0]
        total_nodes = self.total_density_integration_nodes
        nodes = torch.zeros((batch_size, total_nodes), dtype=torch.float32)
        weights = torch.zeros_like(nodes)

        escape_raw = self._get_escape_peak(energy_m_keV, phi_rad).squeeze(1)

        intervals_idx = 0
        current_offset = 0

        for item in self._energy_range:
            if not isinstance(item, list):
                line_energy = float(item)
                nodes[:, current_offset] = line_energy
                weights[:, current_offset] = 1.0
                current_offset += 1
                continue

            Emin, Emax = item
            size = self._density_integration_nodes[intervals_idx]

            if self._flat_intervals[intervals_idx]:
                n_res, w_res = self._scale_nodes_linear(
                    torch.tensor(Emin, dtype=torch.float32), torch.tensor(Emax, dtype=torch.float32),
                    *self._nodes_bkg_0[intervals_idx]
                )
                nodes[:, current_offset : current_offset + size] = n_res
                weights[:, current_offset : current_offset + size] = w_res
                current_offset += size
                intervals_idx += 1
                continue

            width_params = self._width_tensor[intervals_idx]
            photopeak_offset, _, photopeak_scale_right, _, _ = width_params

            missing_direct = self._get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad,
                                                             photopeak_offset, photopeak_scale_right, 
                                                             self._use_photowidth, self._photofraction, self._epsilonfraction,
                                                             inverse=False).squeeze(1)
            missing_wrong  = self._get_missing_energy_peak(phi_igeo_rad, energy_m_keV, phi_rad,
                                                             photopeak_offset, photopeak_scale_right, 
                                                             self._use_photowidth, self._photofraction, self._epsilonfraction,
                                                             inverse=True).squeeze(1)
            raw_peaks = torch.stack([energy_m_keV.squeeze(1), escape_raw, missing_direct, missing_wrong], dim=1)

            interval_peaks = torch.where((raw_peaks >= Emin) & (raw_peaks <= Emax), raw_peaks,
                                          torch.tensor(float('nan'), dtype=torch.float32))

            delta_left = torch.empty_like(interval_peaks)
            delta_right = torch.empty_like(interval_peaks)
            for col, t in enumerate([self._TYPE_P, self._TYPE_E, self._TYPE_M, self._TYPE_WM]):
                dl, dr = self._peak_half_width(t, interval_peaks[:, col], width_params)
                delta_left[:, col] = dl
                delta_right[:, col] = dr

            n_peaks = torch.sum(~torch.isnan(interval_peaks), dim=1)
            over_full = n_peaks > 3
            if torch.any(over_full):
                logger.warning(f"{int(over_full.sum())} events have 4 simultaneously valid peak "
                               f"candidates for this interval.")

            for mode in range(4):
                indices = torch.where(n_peaks == mode)[0]
                if len(indices) == 0:
                    continue
                has_photopeak = ~torch.isnan(interval_peaks[:, 0][indices])

                if mode == 0:
                    Em_sub = raw_peaks[indices, 0] - delta_left[indices, 0]
                    forbidden = Em_sub > Emax
                    valid_indices = indices[~forbidden]
                    if len(valid_indices) > 0:
                        self._fill_nodes(nodes, weights, valid_indices, 0, has_photopeak[~forbidden],
                                          None, None, None, None, intervals_idx, current_offset, Emin, Emax)
                    forbidden_indices = indices[forbidden]
                    if len(forbidden_indices) > 0:
                        nodes[forbidden_indices, current_offset : current_offset + size] = Emin
                    continue

                p_row  = interval_peaks[indices]
                dl_row = delta_left[indices]
                dr_row = delta_right[indices]

                sorted_vals, sorted_type_all = torch.sort(p_row, dim=1)
                sorted_type = sorted_type_all[:, :mode]
                p_sorted  = sorted_vals[:, :mode]
                dl_sorted = torch.gather(dl_row, 1, sorted_type)
                dr_sorted = torch.gather(dr_row, 1, sorted_type)

                self._fill_nodes(nodes, weights, indices, mode, has_photopeak,
                                 p_sorted, dl_sorted, dr_sorted, sorted_type,
                                 intervals_idx, current_offset, Emin, Emax)

            current_offset += size
            intervals_idx += 1

        return nodes, weights
    
    def _get_CDS_coordinates(self, lon_src_rad: torch.Tensor, lat_src_rad: torch.Tensor, indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_lat_src = torch.cos(lat_src_rad)
        sin_lat_src = torch.sin(lat_src_rad)
        cos_lon_src = torch.cos(lon_src_rad)
        sin_lon_src = torch.sin(lon_src_rad)
        
        if indices is None:
            indices = slice(0, len(self._cos_lat_scatt))
        
        cos_geo = (
            cos_lat_src * cos_lon_src * self._cos_lat_scatt[indices] * self._cos_lon_scatt[indices] +
            cos_lat_src * sin_lon_src * self._cos_lat_scatt[indices] * self._sin_lon_scatt[indices] +
            sin_lat_src * self._sin_lat_scatt[indices]
        )
        
        cos_geo = torch.clip(cos_geo, -1.0, 1.0)
        phi_geo_rad = torch.arccos(cos_geo)
        
        return phi_geo_rad, np.pi - phi_geo_rad
    
    def _compute_nodes(self):
        sc_coord_sph = self._sc_coord_sph_cache[self._valid_mask_cache]
        
        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        
        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad), indices=self._valid_mask_cache)
        np_memory_dtype = np.float32 if self._reduce_memory else np.float64
        self._irf_energy_node_cache = np.asarray(self._get_nodes(self._energy_m_keV[self._valid_mask_cache], self._phi_rad[self._valid_mask_cache], phi_geo_rad, phi_igeo_rad)[0], dtype=np_memory_dtype)
    
    def _compute_density_helper(self, indices: np.ndarray, source_coord,
                                sc_coord_sph=None, earth_occ_index: Optional[np.ndarray]=None,
                                buffer: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]=None) -> Tuple[torch.Tensor, np.ndarray]:
        indices = np.asarray(indices)
            
        if sc_coord_sph is None:
            sc_coord_sph = self._get_target_in_sc_frame(source_coord, self._sc_ori_unique)[self._inv_idx[indices]]
        else:
            sc_coord_sph = sc_coord_sph[indices]
        if earth_occ_index is None:
            earth_occ_index = self._earth_occ(source_coord, self._sc_ori_unique)[self._inv_idx[indices]]
        else:
            earth_occ_index = earth_occ_index[indices]
        
        live_sub = self._livetime_ratio[indices]
        current_n = len(earth_occ_index)
        
        e_sl = self._energy_m_keV[indices]
        p_sl = self._phi_rad[indices]

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        
        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad), indices=indices)
        
        nodes, weights = self._get_nodes(e_sl, p_sl, phi_geo_rad, phi_igeo_rad)
        n_energy = self.total_density_integration_nodes
        
        current_total = current_n * n_energy
        
        if buffer is not None:
            batch_lon_src_buffer, batch_lat_src_buffer, batch_energy_buffer, batch_phi_buffer, batch_lon_scatt_buffer, batch_lat_scatt_buffer = buffer
        else:
            buffer_size = n_energy * current_n
            batch_lon_src_buffer   = np.empty(buffer_size, dtype=np.float32)
            batch_lat_src_buffer   = np.empty(buffer_size, dtype=np.float32)
            batch_energy_buffer    = np.empty(buffer_size, dtype=np.float32)
            batch_phi_buffer       = np.empty(buffer_size, dtype=np.float32)
            batch_lon_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lat_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
        
        batch_lon_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lon_ph_rad[:, np.newaxis]
        batch_lat_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lat_ph_rad[:, np.newaxis]
        
        batch_energy_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(e_sl[:, np.newaxis])
        batch_lon_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lon_scatt[indices, np.newaxis])
        batch_lat_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lat_scatt[indices, np.newaxis])
        batch_phi_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(p_sl[:, np.newaxis])
        
        photons = PhotonListWithDirectionAndEnergyInSCFrame(
            batch_lon_src_buffer[:current_total],
            batch_lat_src_buffer[:current_total],
            np.asarray(nodes).ravel()
        )

        events = EmCDSEventDataInSCFrameFromArrays(
            batch_energy_buffer[:current_total],
            batch_lon_scatt_buffer[:current_total],
            batch_lat_scatt_buffer[:current_total],
            batch_phi_buffer[:current_total],
        )
        
        res_block = torch.as_tensor(asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float32)).view(current_n, n_energy)
        
        occ = torch.as_tensor(earth_occ_index).view(-1, 1)
        live = torch.as_tensor(live_sub).view(-1, 1)
        
        res_block *= occ * live * weights
        
        np_memory_dtype = np.float32 if self._reduce_memory else np.float64
        return res_block, np.asarray(nodes, dtype=np_memory_dtype)
    
    def _compute_density(self):
        coord = self._source.position.sky_coord
        n_energy = self.total_density_integration_nodes
        batch_size_events = self._cache_batch_size // n_energy
        
        torch_memory_dtype = torch.float32 if self._reduce_memory else torch.float64
        np_memory_dtype = np.float32 if self._reduce_memory else np.float64
        
        sc_coord_sph = self._sc_coord_sph_cache
        earth_occ_index = self._earth_occ(coord, self._sc_ori_unique)[self._inv_idx]
        self._valid_mask_cache = np.where((earth_occ_index > 0) & (self._livetime_ratio > 0))[0]
        self._valid_events = int(len(self._valid_mask_cache))
        self._valid_mask_tensor = torch.as_tensor(self._valid_mask_cache, dtype=torch.long)
        
        self._irf_cache = torch.zeros((self._valid_events, n_energy), dtype=torch_memory_dtype)
        
        batched_node_caching = (batch_size_events < self._valid_events) & (self._force_energy_node_caching)
        if batched_node_caching:
            self._irf_energy_node_cache = np.zeros((self._valid_events, n_energy), dtype=np_memory_dtype)
        
        #
        #lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        #lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        #
        #phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad))
        #
        buffer_size = n_energy * min(batch_size_events, self._valid_events)
        batch_lon_src_buffer   = np.empty(buffer_size, dtype=np.float32)
        batch_lat_src_buffer   = np.empty(buffer_size, dtype=np.float32)
        batch_energy_buffer    = np.empty(buffer_size, dtype=np.float32)
        batch_phi_buffer       = np.empty(buffer_size, dtype=np.float32)
        batch_lon_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
        batch_lat_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
        
        buffer = (batch_lon_src_buffer, batch_lat_src_buffer, batch_energy_buffer, batch_phi_buffer, batch_lon_scatt_buffer, batch_lat_scatt_buffer)
        
        for i in tqdm(range(0, self._valid_events, batch_size_events), 
                      disable=(not self.show_progress),
                      desc="Caching the response", 
                      smoothing=0.2, 
                      leave=False):
            start = i
            end = min(i + batch_size_events, self._valid_events)
            
            slice_sub = slice(start, end)
            
            res_block, nodes = self._compute_density_helper(self._valid_mask_cache[slice_sub], coord, sc_coord_sph, earth_occ_index, buffer)
            
            #current_n = end - start
            #current_total = current_n * n_energy
            #
            #e_sl = self._energy_m_keV[start:end]
            #p_sl = self._phi_rad[start:end]
            #pg_sl = phi_geo_rad[start:end]
            #pig_sl = phi_igeo_rad[start:end]
            #
            #nodes, weights = self._get_nodes(e_sl, p_sl, pg_sl, pig_sl)

            if batch_size_events >= self._valid_events:
                self._irf_energy_node_cache = nodes.astype(np_memory_dtype)
            if batched_node_caching:
                self._irf_energy_node_cache[start:end] = nodes.astype(np_memory_dtype)
                
            self._irf_cache[start:end] = res_block.to(torch_memory_dtype)
            
            #batch_lon_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lon_ph_rad[start:end, np.newaxis]
            #batch_lat_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lat_ph_rad[start:end, np.newaxis]
            #
            #batch_energy_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._energy_m_keV[start:end, np.newaxis])
            #batch_lon_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lon_scatt[start:end, np.newaxis])
            #batch_lat_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lat_scatt[start:end, np.newaxis])
            #batch_phi_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._phi_rad[start:end, np.newaxis])
            #
            #photons = PhotonListWithDirectionAndEnergyInSCFrame(
            #    batch_lon_src_buffer[:current_total],
            #    batch_lat_src_buffer[:current_total],
            #    np.asarray(nodes).ravel()
            #    )
            #
            #events = EmCDSEventDataInSCFrameFromArrays(
            #    batch_energy_buffer[:current_total],
            #    batch_lon_scatt_buffer[:current_total],
            #    batch_lat_scatt_buffer[:current_total],
            #    batch_phi_buffer[:current_total],
            #)
            
            #res_block = torch.as_tensor(asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float32)).view(current_n, n_energy)
            
            ###eff_areas_flat = torch.as_tensor(asarray(self._irf._effective_area_cm2(photons), dtype=np.float32))
            ###densities_flat = torch.as_tensor(asarray(self._irf._event_probability(photons, events), dtype=np.float32))

            ###res_block = (densities_flat * eff_areas_flat).view(current_n, n_energy)

            #occ = torch.as_tensor(earth_occ_index[start:end]).view(-1, 1)
            #live = torch.as_tensor(self._livetime_ratio[start:end]).view(-1, 1)
            
            #res_block *= occ * live * weights

            #self._irf_cache[start:end] = res_block
    
    def _update_cache(self):
        
        if self._source is None:
            raise RuntimeError("Call set_source() first.")
        
        source_coord = self._source.position.sky_coord
        
        if (self._sc_coord_sph_cache is None) or (source_coord != self._last_convolved_source_skycoord):
            self._sc_coord_sph_cache = self._get_target_in_sc_frame(source_coord, self._sc_ori_unique)[self._inv_idx]
        
        no_recalculation = ((source_coord == self._last_convolved_source_skycoord)
                            and
                            (self._irf_cache is not None)
                            and
                            (self._area_cache is not None))
        
        area_recalculation = ((source_coord != self._last_convolved_source_skycoord)
                              or
                              (self._area_cache is None))
        
        pdf_recalculation = ((source_coord != self._last_convolved_source_skycoord)
                             or
                             (self._irf_cache is None))
        
        if no_recalculation:
            pass
        else:
            active_pool = True
            if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
                active_pool = self._irf.active_pool
                if not active_pool:
                    self._irf.init_compute_pool()
            
            if source_coord != self._last_convolved_source_skycoord:
                self._irf_energy_node_cache = None
            
            if area_recalculation:
                self._compute_area()
                
            if pdf_recalculation:
                self._init_node_pool()
                self._compute_density()
            
            if not active_pool:
                self._irf.shutdown_compute_pool()
            
            self._last_convolved_source_skycoord = source_coord.copy()
        
        node_caching = (self.force_energy_node_caching
                        and
                        self._irf_energy_node_cache is None)
        
        if node_caching:
            self._compute_nodes()
    
    def cache_to_file(self, filename: Union[str, Path]): 
        with h5py.File(str(filename), 'w') as f:
            def processed_energy_range():
                e_range = []
                for x in self._energy_range:
                    if isinstance(x, list):
                        e_range.append(x)
                    else:
                        e_range.append([x, x])
                return np.array(e_range, dtype=np.float64)
            
            f.attrs['total_expectation_resolution'] = self._total_expectation_resolution
            f.attrs['peak_width_params'] = self._peak_width_params
            f.attrs['cache_batch_size'] = self._cache_batch_size
            f.attrs['integration_batch_size'] = self._integration_batch_size
            f.attrs['show_progress'] = self._show_progress
            f.attrs['force_energy_node_caching'] = self._force_energy_node_caching
            f.attrs['reduce_memory'] = self._reduce_memory
            f.attrs['n_intervals'] = self._n_intervals
            
            f.create_dataset('energy_range', data=processed_energy_range())
            f.create_dataset('peak_nodes', data=np.array(self._peak_nodes, dtype=np.int32))
            f.create_dataset('density_integration_nodes', data=np.array(self._density_integration_nodes, dtype=np.int32))
            
            if self._offset is not None:
                f.attrs['offset'] = self._offset
            
            if self._valid_events is not None:
                f.attrs['valid_events'] = self._valid_events
            
            if self._irf_cache is not None:
                f.create_dataset('irf_cache', data=self._irf_cache.numpy(), 
                               compression='gzip')
            
            if self._irf_energy_node_cache is not None:
                f.create_dataset('irf_energy_node_cache', data=self._irf_energy_node_cache,
                               compression='gzip')

            if self._area_cache is not None:
                f.create_dataset('area_cache', data=self._area_cache,
                               compression='gzip')
                
            if self._area_energy_node_cache is not None:
                f.create_dataset('area_energy_node_cache', data=self._area_energy_node_cache,
                               compression='gzip')

            if self._exp_events is not None:
                f.create_dataset('exp_events', data=self._exp_events)

            if self._exp_density is not None:
                f.create_dataset('exp_density', data=self._exp_density.numpy(),
                               compression='gzip')
            
            if self._valid_mask_cache is not None:
                f.create_dataset('valid_mask_cache', data=self._valid_mask_cache,
                               compression='gzip')

            if self._last_convolved_source_dict_number is not None:
                json_str = json.dumps(self._last_convolved_source_dict_number)
                f.attrs['last_convolved_source_dict_number'] = json_str
            
            if self._last_convolved_source_dict_density is not None:
                json_str = json.dumps(self._last_convolved_source_dict_density)
                f.attrs['last_convolved_source_dict_density'] = json_str

            if self._last_convolved_source_skycoord is not None:
                sc = self._last_convolved_source_skycoord
                f.attrs['last_convolved_lon_deg'] = sc.spherical.lon.deg
                f.attrs['last_convolved_lat_deg'] = sc.spherical.lat.deg
                f.attrs['last_convolved_frame'] = sc.frame.name
                if hasattr(sc, 'equinox') and sc.equinox is not None:
                    f.attrs['last_convolved_equinox'] = sc.equinox.value
    
    def cache_from_file(self, filename: Union[str, Path]):
        if not os.path.exists(str(filename)):
            raise FileNotFoundError(f"Cache file {str(filename)} not found.")

        with h5py.File(str(filename), 'r') as f:
            def processed_energy_range(input):
                e_range = []
                for x in input:
                    if (x[0] != x[1]):
                        e_range.append([float(x[0]), float(x[1])])
                    else:
                        e_range.append(float(x[0]))
                return e_range
            
            self._total_expectation_resolution = float(f.attrs['total_expectation_resolution'])
            self._peak_width_params = tuple(f.attrs['peak_width_params'])
            self._cache_batch_size = int(f.attrs['cache_batch_size'])
            self._integration_batch_size = int(f.attrs['integration_batch_size'])
            self._show_progress = bool(f.attrs['show_progress'])
            self._force_energy_node_caching = bool(f.attrs['force_energy_node_caching'])
            self._reduce_memory = bool(f.attrs['reduce_memory'])
            self._n_intervals = int(f.attrs['n_intervals'])
            self._density_integration_nodes = f['density_integration_nodes'][:].tolist()
            self._energy_range = processed_energy_range(f['energy_range'][:].tolist())
            self._peak_nodes = f['peak_nodes'][:].tolist()
            
            if 'offset' in f.attrs:
                self._offset = f.attrs['offset']
            else:
                self._offset = None
            
            if 'valid_events' in f.attrs:
                self._valid_events = int(f.attrs['valid_events'])
            else:
                self._valid_events = None
            
            if 'irf_cache' in f:
                self._irf_cache = torch.from_numpy(f['irf_cache'][:])
            else:
                self._irf_cache = None

            if 'irf_energy_node_cache' in f:
                self._irf_energy_node_cache = f['irf_energy_node_cache'][:]
            else:
                self._irf_energy_node_cache = None

            if 'area_cache' in f:
                self._area_cache = f['area_cache'][:]
            else:
                self._area_cache = None

            if 'area_energy_node_cache' in f:
                self._area_energy_node_cache = f['area_energy_node_cache'][:]
            else:
                self._area_energy_node_cache = None

            if 'exp_events' in f:
                self._exp_events = float(f['exp_events'][()])
            else:
                self._exp_events = None

            if 'exp_density' in f:
                self._exp_density = torch.from_numpy(f['exp_density'][:])
            else:
                self._exp_density = None
            
            if 'valid_mask_cache' in f:
                self._valid_mask_cache = np.asarray(f['valid_mask_cache'][:])
                self._valid_mask_tensor = torch.as_tensor(self._valid_mask_cache, dtype=torch.long)
            else:
                self._valid_mask_cache = None
                self._valid_mask_tensor = None

            if 'last_convolved_source_dict_number' in f.attrs:
                self._last_convolved_source_dict_number = json.loads(f.attrs['last_convolved_source_dict_number'])
            else:
                self._last_convolved_source_dict_number = None
            
            if 'last_convolved_source_dict_density' in f.attrs:
                self._last_convolved_source_dict_density = json.loads(f.attrs['last_convolved_source_dict_density'])
            else:
                self._last_convolved_source_dict_density = None

            if 'last_convolved_lon_deg' in f.attrs:
                lon = f.attrs['last_convolved_lon_deg']
                lat = f.attrs['last_convolved_lat_deg']
                frame = f.attrs['last_convolved_frame']
                equinox = f.attrs.get('last_convolved_equinox', None)

                self._last_convolved_source_skycoord = SkyCoord(lon, lat, unit='deg', frame=frame, equinox=equinox)
            else:
                self._last_convolved_source_skycoord = None
            
            if self._irf_cache is not None:
                self._init_node_pool()

    def _compute_density_for_indices(self, indices: np.ndarray, sc_coord_sph, earth_occ_index, coord) -> np.ndarray:
        self._init_node_pool()
        
        active_pool = True
        if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
            active_pool = self._irf.active_pool
            if not active_pool:
                self._irf.init_compute_pool()

        res_block, nodes = self._compute_density_helper(
            indices, coord, sc_coord_sph=sc_coord_sph, earth_occ_index=earth_occ_index, buffer=None
        )
        
        if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction) and not active_pool:
            self._irf.shutdown_compute_pool()
            
        flux = torch.as_tensor(
            self._source(np.asarray(nodes, dtype=np.float64).ravel()),
            dtype=torch.float64
        ).view(nodes.shape)
        
        exp_density = torch.zeros(len(indices), dtype=torch.float64)
        torch.linalg.vecdot(res_block.to(torch.float64), flux, dim=1, out=exp_density)
        
        result = np.asarray(exp_density, dtype=np.float64)
        if self._offset is not None:
            result += self._offset
            
        return result

    def _integrate_single_event_density(self, event_idx: int, occ_val: float, live_val: float, 
                                        lon_val: float, lat_val: float, 
                                        relerr: float, abserr: float, maxEval: int) -> tuple[float, bool]: 
        from cubature import cubature
        
        if np.isclose(0.0, occ_val * live_val):
            return 0.0, True

        e_meas = self._energy_m_keV[event_idx].item()
        lon_sc = self._lon_scatt[event_idx].item()
        lat_sc = self._lat_scatt[event_idx].item()
        phi_m = self._phi_rad[event_idx].item()

        def density_integrand(Ei):
            energies = Ei[:, 0]
            num_eval = len(energies)
            
            lon_src = np.full(num_eval, lon_val, dtype=np.float32)
            lat_src = np.full(num_eval, lat_val, dtype=np.float32)
            e_meas_arr = np.full(num_eval, e_meas, dtype=np.float32)
            lon_sc_arr = np.full(num_eval, lon_sc, dtype=np.float32)
            lat_sc_arr = np.full(num_eval, lat_sc, dtype=np.float32)
            phi_m_arr = np.full(num_eval, phi_m, dtype=np.float32)

            photons = PhotonListWithDirectionAndEnergyInSCFrame(lon_src, lat_src, energies.astype(np.float32))
            events = EmCDSEventDataInSCFrameFromArrays(e_meas_arr, lon_sc_arr, lat_sc_arr, phi_m_arr)

            diff_area = asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float64)
            flux = asarray(self._source(energies), dtype=np.float64)
            return diff_area * occ_val * live_val * flux

        total_res_val = 0.0
        total_err = 0.0
        
        for x in self._energy_range:
            if isinstance(x, list):
                res, err = cubature(
                    density_integrand, ndim=1, fdim=1,
                    xmin=[x[0]], xmax=[x[1]],
                    vectorized=True, relerr=relerr, abserr=abserr, maxEval=maxEval
                )
                total_res_val += res[0]
                total_err += err[0]
            else:
                Ei = np.array([[float(x)]], dtype=np.float64)
                val = density_integrand(Ei)
                total_res_val += val[0]
        
        allowed_err = max(abserr, relerr * abs(total_res_val))
        return total_res_val, bool(total_err <= allowed_err)

    def _integrate_total_counts(self, coord, relerr: float, abserr: float, maxEval: int) -> tuple[float, bool]:
        from cubature import cubature
        
        sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_center)
        lon_ph_rad_center = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad_center = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        earth_occ_center = self._earth_occ(coord, self._sc_ori_center)
        combined_time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(np.float32) * earth_occ_center

        def total_counts_integrand(Ei):
            energies = Ei[:, 0]
            num_eval = len(energies)
            total_counts_for_energies = np.zeros(num_eval, dtype=np.float64)
            
            for i_e, E in enumerate(energies):
                batch_energies = np.full(len(lon_ph_rad_center), E, dtype=np.float32)
                photons = PhotonListWithDirectionAndEnergyInSCFrame(lon_ph_rad_center, lat_ph_rad_center, batch_energies)
                eff_areas = asarray(self._irf.effective_area_cm2(photons), dtype=np.float64)
                
                total_area_at_E = np.sum(eff_areas * combined_time_weights)
                flux_at_E = self._source(np.array([E]))[0]
                total_counts_for_energies[i_e] = total_area_at_E * flux_at_E
                
            return total_counts_for_energies

        high_prec_total_counts = 0.0
        total_err_total = 0.0

        for x in self._energy_range:
            if isinstance(x, list):
                res_total, err_total = cubature(
                    total_counts_integrand, ndim=1, fdim=1,
                    xmin=[x[0]], xmax=[x[1]],
                    vectorized=True, relerr=relerr, abserr=abserr, maxEval=maxEval
                )
                high_prec_total_counts += float(res_total[0])
                total_err_total += float(err_total[0])
            else:
                Ei = np.array([[float(x)]], dtype=np.float64)
                val_total = total_counts_integrand(Ei)
                high_prec_total_counts += float(val_total[0])

        allowed_err_total = max(abserr, relerr * abs(high_prec_total_counts))
        return high_prec_total_counts, bool(total_err_total <= allowed_err_total)

    def validate_integration(self, 
                             n_events: int, 
                             relerr: float = 4e-4, 
                             abserr: float = 1e-30, 
                             maxEval: int = 1000000,
                             save_path: Optional[Union[str, Path]] = None,
                             previous_results: Optional[dict] = None) -> dict:
        """
        If previous_results is given (either the dict returned by an earlier call, or one
        reloaded from save_path via json.load), the expensive cubature-based reference
        (sampled events, per-event high-precision densities, and high-precision total
        counts) is reused as-is, and only the fast, parameter-dependent side (the
        node-placement-based "optimized" densities/total counts) is recomputed against
        the class's CURRENT settings. n_events/relerr/abserr/maxEval are ignored in that
        case, since no new reference integration happens.
        """
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        pool_was_active = True
        if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
            pool_was_active = self._irf.active_pool
            if not pool_was_active:
                self._irf.init_compute_pool()

        try:
            coord = self._source.position.sky_coord

            sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_unique)[self._inv_idx]
            earth_occ_index = self._earth_occ(coord, self._sc_ori_unique)[self._inv_idx]
            self._valid_mask_cache = np.where((earth_occ_index > 0) & (self._livetime_ratio > 0))[0]
            self._valid_events = int(len(self._valid_mask_cache))

            if previous_results is not None:
                de = previous_results["density_errors"]
                if "high_prec_densities" not in de:
                    raise ValueError("previous_results has no 'high_prec_densities' -- it was produced "
                                      "before this field existed. Run validate_integration once without "
                                      "previous_results to generate a reusable report.")
                sampled_indices = np.asarray(de["sampled_indices"])
                converged_mask = np.asarray(de["converged_mask"], dtype=bool)
                high_prec_densities = np.asarray(de["high_prec_densities"], dtype=np.float64)
                if "high_precision" not in previous_results["total_counts"]:
                    high_prec_total_counts, total_counts_converged = self._integrate_total_counts(coord, relerr, abserr, maxEval)
                else:
                    high_prec_total_counts = previous_results["total_counts"]["high_precision"]
            else:
                sampled_indices = np.random.choice(self._valid_mask_cache, size=min(n_events, self._valid_events), replace=False)

                sc_coord_sph_sampled = sc_coord_sph[sampled_indices]
                earth_occ_sampled = earth_occ_index[sampled_indices]
                livetime_ratio_sampled = self._livetime_ratio[sampled_indices]

                high_prec_densities = []
                converged_mask = []

                for idx, event_idx in enumerate(tqdm(sampled_indices, desc="Computing high-precision densities", disable=not self.show_progress)):
                    res_val, is_converged = self._integrate_single_event_density(
                        event_idx, earth_occ_sampled[idx], livetime_ratio_sampled[idx],
                        sc_coord_sph_sampled[idx].lon.rad, sc_coord_sph_sampled[idx].lat.rad,
                        relerr, abserr, maxEval
                    )
                    if self._offset is not None:
                        res_val += self._offset
                    high_prec_densities.append(res_val)
                    converged_mask.append(is_converged)

                high_prec_densities = np.array(high_prec_densities, dtype=np.float64)
                converged_mask = np.array(converged_mask, dtype=bool)

                if sum(converged_mask) / len(converged_mask) < 0.9:
                    raise RuntimeError("Less than 90% of high-precision densities converged. Try increasing the maximum number of evaluations or the acceptable error.")

                high_prec_total_counts, total_counts_converged = self._integrate_total_counts(coord, relerr, abserr, maxEval)
                if not total_counts_converged:
                    raise RuntimeError("High-precision total counts integration did not converge. Try increasing the maximum number of evaluations or the acceptable error.")

            # --- always recomputed fresh against the class's current parameters ---
            opt_densities = self._compute_density_for_indices(sampled_indices, sc_coord_sph, earth_occ_index, coord)

            if (self._area_cache is None) or (coord != self._last_convolved_source_skycoord):
                self._compute_area()
            flux_area = self._source(self._area_energy_node_cache)
            opt_total_counts = float(np.sum(self._area_cache * flux_area, dtype=float))

            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = (opt_densities / high_prec_densities) - 1.0
                rel_errors = np.nan_to_num(rel_errors, nan=0.0, posinf=0.0, neginf=0.0)

            rel_deviation_total = (opt_total_counts / high_prec_total_counts) - 1.0
            converged_rel_errors = rel_errors[converged_mask]

            if len(converged_rel_errors) > 0:
                mean_err = float(np.mean(converged_rel_errors))
                median_err = float(np.median(converged_rel_errors))
                std_err = float(np.std(converged_rel_errors, ddof=1))
            else:
                mean_err = median_err = std_err = float('nan')

            results = {
                "total_counts": {
                    "optimized": opt_total_counts,
                    "high_precision": high_prec_total_counts,
                    "relative_deviation": rel_deviation_total,
                },
                "density_errors": {
                    "mean": mean_err,
                    "median": median_err,
                    "std": std_err,
                    "raw_relative_errors": rel_errors,
                    "sampled_indices": sampled_indices,
                    "converged_mask": converged_mask,
                    "num_converged": int(np.sum(converged_mask)),
                    "total_sampled": len(sampled_indices),
                    "high_prec_densities": high_prec_densities,
                }
            }

            if save_path is not None:
                serializable_results = {
                    "total_counts": results["total_counts"],
                    "density_errors": {
                        "mean": results["density_errors"]["mean"],
                        "median": results["density_errors"]["median"],
                        "std": results["density_errors"]["std"],
                        "num_converged": results["density_errors"]["num_converged"],
                        "total_sampled": results["density_errors"]["total_sampled"],
                        "raw_relative_errors": results["density_errors"]["raw_relative_errors"].tolist(),
                        "sampled_indices": results["density_errors"]["sampled_indices"].tolist(),
                        "converged_mask": results["density_errors"]["converged_mask"].tolist(),
                        "high_prec_densities": results["density_errors"]["high_prec_densities"].tolist(),
                    }
                }
                with open(str(save_path), 'w') as f:
                    json.dump(serializable_results, f, indent=4)

            return results
            
        finally:
            if not pool_was_active and isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
                self._irf.shutdown_compute_pool()

    @staticmethod
    def report_and_plot_validation(validation_results: Union[dict, str, Path], 
                                   save_path: Optional[Union[str, Path]] = None,
                                   percentile: float = 1.0):
        if isinstance(validation_results, (str, Path)):
            if not os.path.exists(str(validation_results)):
                raise FileNotFoundError(f"Validation file {str(validation_results)} not found.")
            with open(str(validation_results), 'r') as f:
                validation_results = json.load(f)

        tc = validation_results["total_counts"]
        de = validation_results["density_errors"]

        print("=" * 55)
        print("         INTEGRATION VALIDATION REPORT")
        print("=" * 55)
        print(f"Total Expected Counts (Optimized):  {tc['optimized']:.2f}")
        print(f"Total Expected Counts (Reference): {tc['high_precision']:.2f}")
        print(f"Relative Deviation:   {tc['relative_deviation']*100:.1e}%")
        print("-" * 55)
        print("Event Density Relative Errors:")
        print(f"  Convergence:  {de['num_converged']/de['total_sampled']*100:.2f}% of all sampled events converged")
        print(f"  Mean Error:   {de['mean']*100:.1e}%")
        print(f"  Median Error: {de['median']*100:.1e}%")
        print(f"  Std Dev:      {de['std']*100:.1e}%")
        print("=" * 55)

        errors = np.array(de["raw_relative_errors"]) * 100

        if "converged_mask" in de:
            converged_mask = np.array(de["converged_mask"], dtype=bool)
        else:
            converged_mask = np.ones(len(errors), dtype=bool)

        converged_errors = errors[converged_mask]

        if len(converged_errors) == 0:
            raise RuntimeError("Zero events converged successfully.")

        fig, ax = plt.subplots(figsize=(9, 6))
        
        p1, p99 = np.percentile(converged_errors, [percentile, 100-percentile])
        plot_range = (p1, p99) if p1 < p99 else (converged_errors.min(), converged_errors.max())
        
        ax.hist(converged_errors, bins='fd', range=plot_range, alpha=0.75, color='royalblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        ax.axvline(de['median']*100, color='darkorange', linestyle='-', 
                   label=f"Median ({de['median']*100:.1e}%)")
        ax.set_xlabel("Relative Error (%)", fontsize=11)

        ax.set_title("Distribution of Relative Errors in Expectation Densities", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Events", fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
            
        plt.show()
    
    def expected_counts(self) -> float:
        """
        Return the total expected counts.
        """
        self._update_cache()
        source_dict = self._source.to_dict()
        
        if (source_dict != self._last_convolved_source_dict_number) or (self._exp_events is None):
            area = self._area_cache
            flux = self._source(self._area_energy_node_cache)
            self._exp_events = np.sum(area * flux, dtype=float)
            
        self._last_convolved_source_dict_number = source_dict
        return self._exp_events
    
    def expectation_density(self) -> Iterable[float]:
        """
        Return the expected number of counts density. This equals the event probabiliy times the number of events.
        """
        
        self._update_cache()
        source_dict = self._source.to_dict()
        if (source_dict != self._last_convolved_source_dict_density) or (self._exp_density is None):
            self._exp_density = torch.zeros(self._n_events, dtype=torch.float64)
            
            n_energy = self.total_density_integration_nodes
            batch_size = self._integration_batch_size // n_energy

            if (self._irf_energy_node_cache is not None) & (batch_size >= self._valid_events):
                flux = torch.as_tensor(
                    self._source(
                        np.asarray(self._irf_energy_node_cache, dtype=np.float64).ravel()
                        ),
                    dtype=torch.float64
                ).view(self._irf_energy_node_cache.shape)
                
                cache = torch.as_tensor(self._irf_cache, dtype=torch.float64)

                #self._exp_density[self._valid_mask_cache] = torch.linalg.vecdot(cache, flux, dim=1)
                self._exp_density.index_copy_(0, self._valid_mask_tensor, torch.linalg.vecdot(cache, flux, dim=1))

            else:
                if self._irf_energy_node_cache is None:
                    sc_coord_sph = self._sc_coord_sph_cache[self._valid_mask_cache]

                    lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
                    lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

                    phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad), indices=self._valid_mask_cache)

                for i in range(0, self._valid_events, batch_size):
                    end = min(i + batch_size, self._valid_events)

                    if self._irf_energy_node_cache is None:
                        e_sl = self._energy_m_keV[self._valid_mask_tensor[i:end]]
                        p_sl = self._phi_rad[self._valid_mask_tensor[i:end]]
                        pg_sl = phi_geo_rad[i:end]
                        pig_sl = phi_igeo_rad[i:end]

                        nodes, _ = self._get_nodes(e_sl, p_sl, pg_sl, pig_sl)
                    else:
                        nodes = self._irf_energy_node_cache[i:end]
                        
                    nodes = np.asarray(nodes, dtype=np.float64)

                    flux_batch = torch.as_tensor(
                        self._source(nodes.ravel()),
                        dtype=torch.float64
                    ).view(nodes.shape)
                    
                    cache = torch.as_tensor(self._irf_cache[i:end], dtype=torch.float64)
                    
                    #self._exp_density[self._valid_mask_cache[i:end]] = torch.linalg.vecdot(cache, flux_batch, dim=1)
                    self._exp_density.index_copy_(0, self._valid_mask_tensor[i:end], torch.linalg.vecdot(cache, flux_batch, dim=1))
            
        self._last_convolved_source_dict_density = source_dict
        
        result = np.asarray(self._exp_density, dtype=np.float64)
        
        if self._offset is not None:
            return result + self._offset
        else:
            return result
    
    def extract_missing_energy_events(self, num_samples: int = 5) -> dict:
        """
        Extracts CDS parameters and labeled peak energies (direct vs. inverse solutions)
        across specified epsilon ranges.
        """
        if self._source is None:
            raise RuntimeError("Call set_source() first to set source coordinates.")

        # Compute CDS geometry angles
        coord = self._source.position.sky_coord
        sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_unique)[self._inv_idx]
        lon_ph_rad_arr = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad_arr = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        
        phi_geo_rad, _ = self._get_CDS_coordinates(
            torch.as_tensor(lon_ph_rad_arr), torch.as_tensor(lat_ph_rad_arr)
        )

        # Inverse geometry angle correction: pi - phi_geo
        phi_geo_rad_inv = torch.pi - phi_geo_rad

        # Calculate direct and inverse missing energy peaks
        Ei_direct = self._get_missing_energy_peak(phi_geo_rad, self._energy_m_keV, self._phi_rad, inverse=False)
        Ei_inverse = self._get_missing_energy_peak(phi_geo_rad_inv, self._energy_m_keV, self._phi_rad, inverse=True)
        
        eps = (self._energy_m_keV / Ei_direct) - 1.0

        bins = [
            (-0.02, -0.01),
            (-0.03, -0.02),
            (-0.04, -0.03),
            (-0.05, -0.04),
            (-0.06, -0.05),
            (-0.07, -0.06),
            (-0.08, -0.07),
            (-0.09, -0.08),
            (-0.10, -0.09),
            (-0.11, -0.10),
            (-0.12, -0.11),
            (-0.13, -0.12),
            (-0.14, -0.13),
            (-0.15, -0.14),
        ]

        extracted_events = {}

        for low, high in bins:
            mask = (eps >= low) & (eps < high) & (~torch.isnan(Ei_direct))
            matching_indices = torch.where(mask)[0][:num_samples]

            events_data = []
            for idx in matching_indices:
                em = self._energy_m_keV[idx].item()
                peaks_dict = {}
                
                d_val = Ei_direct[idx].item()
                i_val = Ei_inverse[idx].item()
                
                if not np.isnan(d_val) and d_val > em:
                    peaks_dict["direct"] = d_val
                if not np.isnan(i_val) and i_val > em:
                    peaks_dict["inverse"] = i_val

                events_data.append({
                    "event_index": idx.item(),
                    "lon_ph_rad": float(lon_ph_rad_arr[idx]),
                    "lat_ph_rad": float(lat_ph_rad_arr[idx]),
                    "Em_keV": em,
                    "phi_rad": self._phi_rad[idx].item(),
                    "psi_lon_rad": self._lon_scatt[idx].item(),
                    "chi_lat_rad": self._lat_scatt[idx].item(),
                    "Ei_peaks_keV": peaks_dict,  # Dict mapping solution type -> peak value
                    "phi_geo_rad": phi_geo_rad[idx].item(),
                    "epsilon": eps[idx].item(),
                })

            extracted_events[f"[{low}, {high}]"] = events_data

        return extracted_events

class UnbinnedThreeMLPointSourceResponseIRFAdaptiveV2(CachedUnbinnedThreeMLSourceResponseInterface):
    
    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf: FarFieldSpectralInstrumentResponseFunctionInterface,
                 sc_history: SpacecraftHistory,
                 show_progress: bool = True,
                 force_energy_node_caching: bool = False,
                 reduce_memory: bool = True):
        """
        Fold the IRF with the point source spectrum by evaluating the IRF at Ei positions 
        adaptively chosen based on characteristic IRF features.
        
        All IRF queries are cached and can be saved to / loaded from a file.
        """
        print("Init start", flush = True)
        
        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._irf = irf
        self._sc_ori = sc_history
        self.show_progress = show_progress
        self.force_energy_node_caching = force_energy_node_caching
        
        # Default parameters for irf energy node placement
        self._total_expectation_resolution = 18.
        self._peak_nodes = [[19, 13, 13],]#[[18, 12, 12],]
        self._bkg_nodes  = [[8, 10, 14],]
        self._bkg_thresholds = [[500.0, 2000.0],]
        # (photopeak_offset, photopeak_scale, escape_width, missing_energy_scale)
        # half-widths: photopeak = sqrt(Ei + photopeak_offset) * photopeak_scale
        #              escape    = escape_width (constant)
        #              missing_energy (both) = Ei * missing_energy_scale
        self._peak_width_params: Tuple[float, float, float, float, float] = (1200., 0.60, 0.60, 80., 0.15)#(2000., 0.50, 1.0, 120., 0.15)
        self._energy_range = [[100., 10_000.],]
        self._n_intervals = 1
        self._cache_batch_size = 1_000_000
        self._integration_batch_size = 1_000_000
        
        self._offset: Optional[float] = sys.float_info.min
        self._TYPE_P, self._TYPE_E, self._TYPE_M, self._TYPE_WM = 0, 1, 2, 3
        self._N_PEAK_TYPES = 4
        self._presence_eps: float = 0.01
        self._center_scale_beta: Tuple[float, float, float] = (2.5, 4.5, 3.0)#(2.5, 3.5, 2.0)
        self._boundary_admit_frac: float = 0.25
        
        # Node Base Pool Placeholders (Tensors)
        self._nodes_peaks_pool: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_bkg_pool: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]] = None
        
        self._peak_patterns: List[Tuple[int, ...]] = self._get_peak_patterns()
        self._pattern_lookup_table: torch.Tensor = self._build_pattern_lookup_table()
        self._bucket_tables: Optional[List[List[dict]]] = None
        self._bucket_key_lookup: Optional[List[torch.Tensor]] = None
        self._bucket_event_indices: Optional[List[Dict[int, torch.Tensor]]] = None
        
        self._width_tensor: List[Tuple[float, float, float, float]] = None
        self._bkg_thresholds_tensor: List[torch.Tensor] = None
        
        # Checks to avoid unecessary recomputations
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None
        
        # Cached values
        self._line_irf_cache: Optional[List[Optional[torch.Tensor]]] = None
        self._bucket_irf_cache: Optional[List[Dict[int, torch.Tensor]]] = None # cm^2/rad/sr
        self._bucket_energy_node_cache: Optional[List[Dict[int, torch.Tensor]]] = None
        self._area_cache: Optional[np.ndarray] = None # cm^2*s*keV
        self._area_energy_node_cache: Optional[np.ndarray] = None
        self._exp_events: Optional[float] = None
        self._exp_density: Optional[torch.Tensor] = None
        self._valid_mask_cache: Optional[np.ndarray] = None
        self._valid_events: Optional[int] = None
        self._valid_mask_tensor: Optional[torch.Tensor] = None
        
        # Bucket Caching structures
        self._bucket_indices_cache: Optional[List[Dict[int, torch.Tensor]]] = None
        
        print("Init time interp start", flush = True)
        
        # Precomputed spacecraft history - Midpoint
        self._mid_times = self._sc_ori.obstime[:-1] + (self._sc_ori.obstime[1:] - self._sc_ori.obstime[:-1]) / 2
        self._sc_ori_center = self._sc_ori.interp(self._mid_times)
        
        # Event time and geometry vectors
        data_times = self._data.time
        self._n_events = self._data.nevents
        self._unique_unix, self._inv_idx = np.unique(data_times.utc.unix, return_inverse=True)
        unique_times_obj = Time(self._unique_unix, format='unix', scale='utc')
        self._sc_ori_unique = self._sc_ori.interp(unique_times_obj)
        
        interval_ratios = (self._sc_ori.livetime.to_value(u.s) / self._sc_ori.intervals_duration.to_value(u.s))
        bin_indices = np.searchsorted(self._sc_ori.obstime.utc.unix, self._unique_unix, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(self._sc_ori.livetime) - 1)
        unique_ratio = interval_ratios[bin_indices]
        self._livetime_ratio = unique_ratio[self._inv_idx].astype(np.float32)
        
        print("Init data loading start", flush = True)
        
        # Event kinematics tensors
        self._energy_m_keV = torch.as_tensor(asarray(self._data.energy_keV, dtype=np.float32))
        self._phi_rad = torch.as_tensor(asarray(self._data.scattering_angle_rad, dtype=np.float32))
        
        self._lon_scatt = torch.as_tensor(asarray(self._data.scattered_lon_rad_sc, dtype=np.float32))
        self._lat_scatt = torch.as_tensor(asarray(self._data.scattered_lat_rad_sc, dtype=np.float32))
        self._cos_lat_scatt = torch.cos(self._lat_scatt)
        self._sin_lat_scatt = torch.sin(self._lat_scatt)
        self._cos_lon_scatt = torch.cos(self._lon_scatt)
        self._sin_lon_scatt = torch.sin(self._lon_scatt)
        
        print("Init prepare integration structures start", flush = True)
        
        # Also runs _check_memory_savings
        self.reduce_memory = reduce_memory
        self._prepare_integration_structures() # TODO: Move to another place
        # TODO: Add getters and setters
    
    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface
    
    @property
    def total_expectation_integration_nodes(self) -> int:
        total = 0
        for x in self._energy_range:
            if isinstance(x, float):
                total += 1
            else:
                total += self._total_expectation_integration_nodes(self._total_expectation_resolution, x[1] - x[0])
        return total
    
    def _total_expectation_integration_nodes(self, resolution: float, diff: float) -> int:
        return int(np.ceil(diff / resolution)) + 1

    def _peak_node_count(self, interval_idx: int, peak_type: int) -> int:
        peak_nodes = self._peak_nodes[interval_idx]
        return peak_nodes[2] if peak_type in (self._TYPE_M, self._TYPE_WM) else peak_nodes[peak_type]
    
    
    def set_source(self, source: Source):
        if not isinstance(source, PointSource):
            raise TypeError("Please provide a PointSource!")

        self._source = source
    
    def copy(self) -> CachedUnbinnedThreeMLSourceResponseInterface:
        new_instance = copy.copy(self)
        new_instance.clear_cache()
        new_instance._source = None
        
        return new_instance
    
    def clear_cache(self):
        # Source-change tracking
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None

        # Valid-event mask (depends on source position via earth occultation)
        self._valid_mask_cache = None
        self._valid_events = None
        self._valid_mask_tensor = None

        # Per-bucket / per-line caches
        self._bucket_event_indices = None
        self._bucket_irf_cache = None
        self._bucket_energy_node_cache = None
        self._line_irf_cache = None

        # Area / total-counts cache
        self._area_cache = None
        self._area_energy_node_cache = None
        self._exp_events = None

        # Density cache
        self._exp_density = None
    
    def init_cache(self):
        self._update_cache()
    
    
    def _get_peak_patterns(self) -> List[Tuple[int, ...]]:
        patterns = [()]
        for n in (1, 2, 3):
            for seq in permutations(range(self._N_PEAK_TYPES), n):
                if self._TYPE_P in seq and seq[0] != self._TYPE_P:
                    continue
                if set(seq) >= {self._TYPE_E, self._TYPE_M, self._TYPE_WM}:
                    continue
                patterns.append(seq)
        return patterns

    def _build_pattern_lookup_table(self) -> torch.Tensor:
        table = torch.full((125,), -1, dtype=torch.long)
        for pid, seq in enumerate(self._peak_patterns):
            padded = seq + (-1,) * (3 - len(seq))
            d0, d1, d2 = (p + 1 for p in padded)
            table[d0 * 25 + d1 * 5 + d2] = pid
        return table

    def _build_bucket_table(self, interval_idx: int) -> Tuple[List[dict], torch.Tensor]:
        print("Build bucket table start", flush = True)
        bkg_nodes = self._bkg_nodes[interval_idx]
        n_sizes = len(bkg_nodes)
        MAX_SLOTS = 3
        KEY_STRIDE_K = n_sizes ** MAX_SLOTS
        KEY_STRIDE_PATTERN = KEY_STRIDE_K * 4

        bucket_table: List[dict] = []
        key_to_id: Dict[int, int] = {}

        for pattern_id, seq in enumerate(self._peak_patterns):
            n = len(seq)
            if n == 0:
                for size in range(n_sizes):
                    bucket_id = len(bucket_table)
                    key = pattern_id * KEY_STRIDE_PATTERN + 0 * KEY_STRIDE_K + size
                    key_to_id[key] = bucket_id
                    bucket_table.append(dict(seq=seq, k=0, sizes=(size,), n_nodes=bkg_nodes[size]))

                bucket_id = len(bucket_table)
                key = pattern_id * KEY_STRIDE_PATTERN + 0 * KEY_STRIDE_K + n_sizes
                key_to_id[key] = bucket_id
                bucket_table.append(dict(seq=(), k=0, sizes=(), n_nodes=0, is_empty=True))
                continue

            M = n - 1
            O = (1 if seq[0] != self._TYPE_P else 0) + 1
            peak_node_total = sum(self._peak_node_count(interval_idx, t) for t in seq)

            for k in range(O + 1):
                n_slots = M + k
                for sizes in product(range(n_sizes), repeat=n_slots):
                    bucket_id = len(bucket_table)
                    node_count = peak_node_total + sum(bkg_nodes[s] for s in sizes)
                    sizes_encoded = 0
                    for s in sizes:
                        sizes_encoded = sizes_encoded * n_sizes + s
                    key = pattern_id * KEY_STRIDE_PATTERN + k * KEY_STRIDE_K + sizes_encoded
                    key_to_id[key] = bucket_id
                    bucket_table.append(dict(seq=seq, k=k, sizes=sizes, n_nodes=node_count))

        max_key = KEY_STRIDE_PATTERN * len(self._peak_patterns)
        key_lookup = torch.full((max_key,), -1, dtype=torch.long)
        for key, bid in key_to_id.items():
            key_lookup[key] = bid
        
        print("Build bucket table end", flush = True)

        return bucket_table, key_lookup

    def _group_by_bucket(self, bucket_id: torch.Tensor, n_buckets: int) -> Dict[int, torch.Tensor]:
        print("Group by bucket start", flush = True)
        
        order = torch.argsort(bucket_id, stable=True)
        sorted_ids = bucket_id[order]
        counts = torch.bincount(sorted_ids, minlength=n_buckets)
        offsets = torch.cumsum(counts, 0) - counts
        
        print("Group by bucket end", flush = True)
        
        return {b: order[offsets[b]:offsets[b] + counts[b]]
                for b in range(n_buckets) if counts[b] > 0}
    
    def _prepare_integration_structures(self):
        """
        Materializes everything _classify_interval needs from the raw parameter
        lists (_peak_width_params, _bkg_thresholds, _peak_nodes, _bkg_nodes), and
        builds the per-interval bucket tables.
        """
        print("Prepare integration structures start", flush = True)
        self._width_tensor: List[Tuple[float, float, float, float]] = [
            tuple(self._peak_width_params) for _ in range(self._n_intervals)
        ]

        self._bkg_thresholds_tensor: List[torch.Tensor] = [
            torch.as_tensor(self._bkg_thresholds[i], dtype=torch.float32)
            for i in range(self._n_intervals)
        ]

        self._bucket_tables = []
        self._bucket_key_lookup = []
        for interval_idx in range(self._n_intervals):
            table, key_lookup = self._build_bucket_table(interval_idx)
            self._bucket_tables.append(table)
            self._bucket_key_lookup.append(key_lookup)
        print("Prepare integration structures end", flush = True)

        self._build_node_pools()
    
    @staticmethod
    def _build_nodes(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, w = np.polynomial.legendre.leggauss(degree)
        return torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), torch.as_tensor(w, dtype=torch.float32).unsqueeze(0)
    
    def _build_node_pools(self):
        """
        Unit Gauss-Legendre nodes/weights on [-1, 1], keyed by node count only --
        a degree-18 pool is identical whether it's used for a photopeak or a large
        background segment, so there's no need to key by type/size, just by n.
        This dedupes automatically (e.g. missing-energy and missing-wrong-order
        peaks sharing the same node count reuse the same pool entry).
        """
        needed_peak_degrees = {n for interval in self._peak_nodes for n in interval}
        needed_bkg_degrees  = {n for interval in self._bkg_nodes for n in interval}

        self._nodes_peaks_pool = {n: self._build_nodes(n) for n in needed_peak_degrees}
        self._nodes_bkg_pool   = {n: self._build_nodes(n) for n in needed_bkg_degrees}
    
    @staticmethod
    def _earth_occ(source_coord: SkyCoord, ori: SpacecraftHistory) -> np.ndarray:
        dist_earth_center = ori.location.spherical.distance.km
        max_angle = np.pi - np.arcsin(c.R_earth.to(u.km).value/dist_earth_center)
        src_angle = source_coord.separation(ori.earth_zenith)
        return (src_angle.to(u.rad).value < max_angle).astype(np.float32)
        
    @staticmethod
    def _get_target_in_sc_frame(source_coord: SkyCoord, ori: SpacecraftHistory) -> SkyCoord:
        src_in_sc_frame = SkyCoord(np.dot(ori.attitude.rot.inv().as_matrix(), source_coord.transform_to(ori.attitude.frame).cartesian.xyz.value),
                                   representation_type = 'cartesian', frame = SpacecraftFrame())

        src_in_sc_frame.representation_type = 'spherical'
        return src_in_sc_frame
    
    def _get_CDS_coordinates(self, lon_src_rad: torch.Tensor, lat_src_rad: torch.Tensor, indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_lat_src = torch.cos(lat_src_rad)
        sin_lat_src = torch.sin(lat_src_rad)
        cos_lon_src = torch.cos(lon_src_rad)
        sin_lon_src = torch.sin(lon_src_rad)
        
        if indices is None:
            indices = slice(0, len(self._cos_lat_scatt))
        
        cos_geo = (
            cos_lat_src * cos_lon_src * self._cos_lat_scatt[indices] * self._cos_lon_scatt[indices] +
            cos_lat_src * sin_lon_src * self._cos_lat_scatt[indices] * self._sin_lon_scatt[indices] +
            sin_lat_src * self._sin_lat_scatt[indices]
        )
        
        cos_geo = torch.clip(cos_geo, -1.0, 1.0)
        phi_geo_rad = torch.arccos(cos_geo)
        
        return phi_geo_rad, np.pi - phi_geo_rad
    
    @staticmethod
    def _scale_nodes_exp(E1: torch.Tensor, E2: torch.Tensor, 
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = E2 - E1

        out_n = (nodes_u + 1).mul(0.5).pow(2).mul(diff).add(E1)
        out_w = (nodes_u + 1).mul(0.5).mul(weights_u).mul(diff)

        return out_n, out_w
    
    @staticmethod
    def _scale_nodes_linear(E1: torch.Tensor, E2: torch.Tensor,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = E2 - E1
 
        out_n = (nodes_u + 1).mul(0.5).mul(diff).add(E1)
        out_w = weights_u.mul(0.5).mul(diff)
 
        return out_n, out_w

    @staticmethod
    def _scale_nodes_center(E1: torch.Tensor, E2: torch.Tensor, EC: torch.Tensor,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor,
                            beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        width_left = (EC - E1)
        width_right = (E2 - EC)
        total_width = width_left + width_right
        
        is_zero = (total_width < 1e-9)
        safe_total = torch.where(is_zero, torch.ones_like(total_width), total_width)

        f = width_left / safe_total
        u0 = 2.0 * f - 1.0
        u0 = torch.clamp(u0, min=-1.0 + 1e-6, max=1.0 - 1e-6) 

        mask_left = (nodes_u < u0)
        t_left  = (nodes_u - u0) / (u0 + 1.0)
        t_right = (nodes_u - u0) / (1.0 - u0)
        t = torch.where(mask_left, t_left, t_right)

        scale = torch.where(mask_left, width_left, width_right)
        jac_denom = torch.where(mask_left, (u0 + 1.0), (1.0 - u0))

        sinh_beta = float(np.sinh(beta))
        out_n = EC + scale * torch.sinh(beta * t) / sinh_beta
        out_w = weights_u * scale * beta * torch.cosh(beta * t) / sinh_beta / jac_denom

        w_sum = out_w.sum(dim=-1, keepdim=True)
        sum_is_zero = (w_sum.abs() < 1e-12)
        safe_w_sum = torch.where(sum_is_zero, torch.ones_like(w_sum), w_sum)
        out_w = out_w * total_width / safe_w_sum

        out_n = torch.where(is_zero | sum_is_zero, EC, out_n)
        out_w = torch.where(is_zero | sum_is_zero, torch.zeros_like(out_w), out_w)

        return out_n, out_w
    
    #def _scale_nodes_center(self, E1: torch.Tensor, E2: torch.Tensor, EC: torch.Tensor,
    #                        nodes_u: torch.Tensor, weights_u: torch.Tensor,
    #                        beta: Optional[float] = None,
    #                        peak_type: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #    """
    #    Denser-near-EC node placement, breakpoint shifted to reflect the actual
    #    width ratio (as before). Escape peaks use the original cubic (power-3)
    #    clustering instead of the sinh transform -- empirically the sinh shape
    #    doesn't suit the escape peak's response well, whereas the photopeak and
    #    missing-energy peaks are fine with it.
    #    """
    #    if beta is None:
    #        beta = self._center_scale_beta
#
    #    width_left = (EC - E1)
    #    width_right = (E2 - EC)
    #    total_width = width_left + width_right
#
    #    is_zero = (total_width < 1e-9)
    #    safe_total = torch.where(is_zero, torch.ones_like(total_width), total_width)
#
    #    f = width_left / safe_total
    #    u0 = 2.0 * f - 1.0
    #    u0 = torch.clamp(u0, min=-1.0 + 1e-6, max=1.0 - 1e-6)
#
    #    mask_left = (nodes_u < u0)
    #    t_left  = (nodes_u - u0) / (u0 + 1.0)
    #    t_right = (nodes_u - u0) / (1.0 - u0)
    #    t = torch.where(mask_left, t_left, t_right)
#
    #    scale = torch.where(mask_left, width_left, width_right)
    #    jac_denom = torch.where(mask_left, (u0 + 1.0), (1.0 - u0))
#
    #    if peak_type == self._TYPE_E:
    #        # escape peak: cubic clustering, same asymmetric breakpoint as everything else
    #        out_n = EC + t.pow(3) * scale
    #        out_w = t.pow(2).mul(3).mul(weights_u).mul(scale).div(jac_denom)
    #    else:
    #        sinh_beta = np.sinh(beta)
    #        out_n = EC + scale * torch.sinh(beta * t) / sinh_beta
    #        out_w = weights_u * scale * beta * torch.cosh(beta * t) / sinh_beta / jac_denom
#
    #    w_sum = out_w.sum(dim=-1, keepdim=True)
    #    sum_is_zero = (w_sum.abs() < 1e-12)
    #    safe_w_sum = torch.where(sum_is_zero, torch.ones_like(w_sum), w_sum)
    #    out_w = out_w * total_width / safe_w_sum
#
    #    out_n = torch.where(is_zero | sum_is_zero, EC, out_n)
    #    out_w = torch.where(is_zero | sum_is_zero, torch.zeros_like(out_w), out_w)
#
    #    return out_n, out_w

    @staticmethod
    def _scale_nodes_log(E1: torch.Tensor, E2: torch.Tensor,
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_E1 = torch.log10(E1)
        log_E2 = torch.log10(E2)
        scale = 0.5 * (log_E2 - log_E1)

        out_n = torch.pow(10, nodes_u.mul(scale).add(0.5 * (log_E1 + log_E2)))
        out_w = out_n.mul(weights_u).mul(scale).mul(torch.log(torch.tensor(10.0)))

        return out_n, out_w
        
    def _get_escape_peak(self, energy_m_keV: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        E2 = 511.0 / (1.0 + 511.0 / energy_m_keV - torch.cos(phi_rad))
        energy = energy_m_keV + 1022.0 - E2
        
        accept = (energy > 1600.0) & (energy_m_keV < energy)
        return torch.where(accept, energy, torch.tensor(float('nan'), dtype=torch.float32))
    
    def _get_missing_energy_peak(self, phi_geo_rad: torch.Tensor, energy_m_keV: torch.Tensor, 
                                 phi_rad: torch.Tensor, photopeak_offset: float, photopeak_scale_right: float,
                                 inverse: bool = False) -> torch.Tensor:
        cos_geo = torch.cos(phi_geo_rad)
        cos_phi = torch.cos(phi_rad)
        
        if inverse:
            denom = 2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            root = torch.sqrt(energy_m_keV * (cos_geo - 1) * (-2044.0 - 5 * energy_m_keV + energy_m_keV * cos_geo + 4 * energy_m_keV * cos_phi))
            energy = 511.0 * (energy_m_keV - energy_m_keV * cos_geo + root) / denom
        else:
            denom = 2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            root = torch.sqrt(energy_m_keV**2 * (cos_geo - 1) * (cos_phi - 1) *
                              ((1022.0 + energy_m_keV)**2 - energy_m_keV * (2044.0 + energy_m_keV) * cos_phi - 2 * energy_m_keV**2 * cos_geo * torch.sin(phi_rad/2)**2))
            energy = (energy_m_keV**2 * (1 - cos_geo - cos_phi + cos_phi * cos_geo) + root) / denom

        hw_P_right = torch.sqrt(energy_m_keV.clamp(min=-photopeak_offset + 1e-6) + photopeak_offset) * photopeak_scale_right
        accept = energy >= (energy_m_keV + hw_P_right)
        return torch.where(accept, energy, torch.tensor(float('nan'), dtype=torch.float32))
    
    def _peak_value(self, peak_type: int, energy_m_keV, phi_rad, phi_geo_rad, phi_igeo_rad,
                    width_params: tuple) -> torch.Tensor:
        photopeak_offset, photopeak_scale_left, photopeak_scale_right, escape_width, missing_energy_scale = width_params
        if peak_type == self._TYPE_P:
            return energy_m_keV
        elif peak_type == self._TYPE_E:
            return self._get_escape_peak(energy_m_keV, phi_rad)
        elif peak_type == self._TYPE_M:
            return self._get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad,
                                                  photopeak_offset, photopeak_scale_right, inverse=False)
        else:  # self._TYPE_WM
            return self._get_missing_energy_peak(phi_igeo_rad, energy_m_keV, phi_rad,
                                                  photopeak_offset, photopeak_scale_right, inverse=True)
    
    def _peak_half_width(self, peak_type: int, peak_val: torch.Tensor, width_params: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        photopeak_offset, photopeak_scale_left, photopeak_scale_right, escape_width, missing_energy_scale = width_params
        if peak_type == self._TYPE_P:
            base = torch.sqrt(peak_val.clamp(min=-photopeak_offset + 1e-6) + photopeak_offset)
            return base * photopeak_scale_left, base * photopeak_scale_right
        elif peak_type == self._TYPE_E:
            w = torch.full_like(peak_val, escape_width)
            return w, w
        else:
            w = peak_val * missing_energy_scale
            return w, w
    
    def _classify_interval(self, energy_m_keV: torch.Tensor, phi_rad: torch.Tensor,
                            phi_geo_rad: torch.Tensor, phi_igeo_rad: torch.Tensor,
                            interval_idx: int, Emin: float, Emax: float,
                            presence_eps: Optional[float] = None,
                            boundary_frac: Optional[float] = None) -> torch.Tensor:
        if presence_eps is None:
            presence_eps = self._presence_eps
        if boundary_frac is None:
            boundary_frac = self._boundary_admit_frac
        
        print("Start classifying", flush = True)

        N = energy_m_keV.shape[0]
        energy_m_keV = energy_m_keV.view(-1, 1)
        phi_rad = phi_rad.view(-1, 1)
        phi_geo_rad = phi_geo_rad.view(-1, 1)
        phi_igeo_rad = phi_igeo_rad.view(-1, 1)
        n_sizes = len(self._bkg_nodes[interval_idx])

        photopeak_offset, photopeak_scale_left, photopeak_scale_right, escape_width, missing_energy_scale = self._width_tensor[interval_idx]

        escape_val = self._get_escape_peak(energy_m_keV, phi_rad)
        missing_true  = self._get_missing_energy_peak(phi_geo_rad,  energy_m_keV, phi_rad,
                                                      photopeak_offset, photopeak_scale_right, inverse=False)
        missing_wrong = self._get_missing_energy_peak(phi_igeo_rad, energy_m_keV, phi_rad,
                                                      photopeak_offset, photopeak_scale_right, inverse=True)

        raw = torch.cat([energy_m_keV, escape_val, missing_true, missing_wrong], dim=1)

        peak_types_by_column = [self._TYPE_P, self._TYPE_E, self._TYPE_M, self._TYPE_WM]

        delta_left  = torch.empty_like(raw)
        delta_right = torch.empty_like(raw)
        for col, t in enumerate(peak_types_by_column):
            dl, dr = self._peak_half_width(t, raw[:, col], self._width_tensor[interval_idx])
            delta_left[:, col]  = dl
            delta_right[:, col] = dr

        valid = ~torch.isnan(raw)
        below = valid & (raw < Emin)
        above = valid & (raw > Emax)
        in_range = valid & (~below) & (~above)

        delta_admit = torch.where(below, delta_right, delta_left)
        d_edge = torch.where(below, Emin - raw, torch.where(above, raw - Emax, torch.zeros_like(raw)))
        admit_boundary = (below | above) & (d_edge <= boundary_frac * delta_admit)

        inf_t = torch.tensor(float('inf'))
        d_below_masked = torch.where(admit_boundary & below, d_edge, inf_t)
        d_above_masked = torch.where(admit_boundary & above, d_edge, inf_t)
        min_d_below, argmin_below = d_below_masked.min(dim=1)
        min_d_above, argmin_above = d_above_masked.min(dim=1)
        col_idx = torch.arange(4).view(1, 4)
        keep_below = (col_idx == argmin_below.view(-1, 1)) & torch.isfinite(min_d_below).view(-1, 1)
        keep_above = (col_idx == argmin_above.view(-1, 1)) & torch.isfinite(min_d_above).view(-1, 1)

        admitted = in_range | (admit_boundary & below & keep_below) | (admit_boundary & above & keep_above)

        sort_val = torch.where(in_range, raw,
                      torch.where(below, torch.full_like(raw, Emin), torch.full_like(raw, Emax)))
        peak_vals = torch.where(admitted, sort_val, torch.tensor(float('nan')))

        sorted_sortval_all, sorted_type_all = torch.sort(peak_vals, dim=1)
        n_raw_valid = (~torch.isnan(sorted_sortval_all)).sum(dim=1)

        if torch.any(n_raw_valid > 3):
            raise ValueError(f"{int((n_raw_valid > 3).sum())} events have 4 simultaneously admitted peak candidates")

        sorted_type = sorted_type_all[:, :3]
        n_peaks = torch.clamp(n_raw_valid, max=3)

        sorted_true_center = torch.gather(raw, 1, sorted_type.clamp(min=0))
        sorted_delta_left  = torch.gather(delta_left,  1, sorted_type.clamp(min=0))
        sorted_delta_right = torch.gather(delta_right, 1, sorted_type.clamp(min=0))

        a = torch.where(n_peaks >= 1, sorted_type[:, 0], torch.tensor(-1))
        b = torch.where(n_peaks >= 2, sorted_type[:, 1], torch.tensor(-1))
        c = torch.where(n_peaks >= 3, sorted_type[:, 2], torch.tensor(-1))
        code = (a + 1) * 25 + (b + 1) * 5 + (c + 1)
        pattern_id = self._pattern_lookup_table[code]

        left_raw  = sorted_true_center - sorted_delta_left
        right_raw = sorted_true_center + sorted_delta_right
        clamped_left  = left_raw.clone()
        clamped_right = right_raw.clone()

        mid01 = (sorted_true_center[:, 0] + sorted_true_center[:, 1]) / 2.0
        mid12 = (sorted_true_center[:, 1] + sorted_true_center[:, 2]) / 2.0

        clamped_right[:, 0] = torch.where(n_peaks >= 2, torch.min(clamped_right[:, 0], mid01), clamped_right[:, 0])
        clamped_left[:, 1]  = torch.where(n_peaks >= 2, torch.max(clamped_left[:, 1], mid01), clamped_left[:, 1])
        clamped_right[:, 1] = torch.where(n_peaks >= 3, torch.min(clamped_right[:, 1], mid12), clamped_right[:, 1])
        clamped_left[:, 2]  = torch.where(n_peaks >= 3, torch.max(clamped_left[:, 2], mid12), clamped_left[:, 2])

        clamped_right = torch.max(clamped_right, clamped_left)
        clamped_left  = clamped_left.clamp(min=Emin, max=Emax)
        clamped_right = clamped_right.clamp(min=Emin, max=Emax)

        is_bkg_only = (n_peaks == 0)
        leading_optional  = (n_peaks > 0) & (sorted_type[:, 0] != self._TYPE_P)
        trailing_optional = (n_peaks > 0)

        leading_width = clamped_left[:, 0] - Emin
        last_idx = (n_peaks - 1).clamp(min=0)
        last_right = torch.gather(clamped_right, 1, last_idx.view(-1, 1)).squeeze(1)
        trailing_width = Emax - last_right

        gap_width = torch.stack([
            clamped_left[:, 1] - clamped_right[:, 0],
            clamped_left[:, 2] - clamped_right[:, 1],
        ], dim=1)

        leading_present  = leading_optional  & (leading_width  > presence_eps)
        trailing_present = trailing_optional & (trailing_width > presence_eps)

        thresholds = self._bkg_thresholds_tensor[interval_idx]
        size_leading  = torch.bucketize(leading_width.clamp(min=0.0), thresholds)
        size_trailing = torch.bucketize(trailing_width.clamp(min=0.0), thresholds)
        size_gap0     = torch.bucketize(gap_width[:, 0].clamp(min=0.0), thresholds)
        size_gap1     = torch.bucketize(gap_width[:, 1].clamp(min=0.0), thresholds)

        photopeak_raw = energy_m_keV.squeeze(1)
        hw_P_left = delta_left[:, 0]
        d_P_above = photopeak_raw - Emax

        empty  = is_bkg_only & (d_P_above > hw_P_left)
        shrunk = is_bkg_only & (d_P_above > 0) & (d_P_above <= hw_P_left)

        whole_width_normal = torch.full((N,), Emax - Emin, dtype=torch.float32)
        whole_width_shrunk = (hw_P_left - d_P_above).clamp(min=0.0)
        whole_width = torch.where(shrunk, whole_width_shrunk, whole_width_normal)
        size_bkg_only = torch.bucketize(whole_width.clamp(min=0.0), thresholds)
        size_bkg_only_final = torch.where(empty, torch.tensor(n_sizes), size_bkg_only)

        M = (n_peaks - 1).clamp(min=0)
        k = leading_present.long() + trailing_present.long()

        slot = torch.full((N, 3), -1, dtype=torch.long)
        slot[:, 0] = torch.where(n_peaks >= 2, size_gap0, slot[:, 0])
        slot[:, 1] = torch.where(n_peaks >= 3, size_gap1, slot[:, 1])

        lead_idx  = M
        trail_idx = M + leading_present.long()
        slot.scatter_(1, lead_idx.view(-1, 1),
                       torch.where(leading_present, size_leading, torch.full_like(size_leading, -1)).view(-1, 1))
        slot.scatter_(1, trail_idx.view(-1, 1),
                       torch.where(trailing_present, size_trailing, torch.full_like(size_trailing, -1)).view(-1, 1))

        n_slots = torch.where(is_bkg_only, torch.tensor(1), M + k)
        slot[:, 0] = torch.where(is_bkg_only, size_bkg_only_final, slot[:, 0])

        sizes_encoded = torch.zeros(N, dtype=torch.long)
        for i in range(3):
            active = i < n_slots
            digit = slot[:, i].clamp(min=0)
            sizes_encoded = torch.where(active, sizes_encoded * n_sizes + digit, sizes_encoded)

        pattern_id_final = torch.where(is_bkg_only, torch.tensor(0, dtype=pattern_id.dtype), pattern_id)
        k_final = torch.where(is_bkg_only, torch.tensor(0), k)

        KEY_STRIDE_K = n_sizes ** 3
        KEY_STRIDE_PATTERN = KEY_STRIDE_K * 4
        key = pattern_id_final * KEY_STRIDE_PATTERN + k_final * KEY_STRIDE_K + sizes_encoded

        bucket_id = self._bucket_key_lookup[interval_idx][key]

        if torch.any(bucket_id < 0):
            n_bad = int((bucket_id < 0).sum())
            raise RuntimeError(f"{n_bad} events produced an invalid bucket key -- "
                                f"classification logic and _build_bucket_table have diverged.")
        
        print("Finished classifying", flush = True)

        return bucket_id
    
    def _fill_bucket_nodes(self, interval_idx: int, bucket_id: int, event_indices: torch.Tensor,
                            Emin: float, Emax: float, presence_eps: Optional[float] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if presence_eps is None:
            presence_eps = self._presence_eps

        record = self._bucket_tables[interval_idx][bucket_id]
        seq, k, sizes, n_nodes = record['seq'], record['k'], record['sizes'], record['n_nodes']
        n_bucket = event_indices.shape[0]
        bkg_nodes = self._bkg_nodes[interval_idx]
        width_params = self._width_tensor[interval_idx]

        if n_nodes == 0:
            return (torch.empty((n_bucket, 0), dtype=torch.float32),
                    torch.empty((n_bucket, 0), dtype=torch.float32))

        nodes_out = torch.empty((n_bucket, n_nodes), dtype=torch.float32)
        weights_out = torch.empty((n_bucket, n_nodes), dtype=torch.float32)

        if len(seq) == 0:
            photopeak_offset, photopeak_scale_left = width_params[0], width_params[1]
            energy_m_keV_bucket = self._energy_m_keV[event_indices].view(-1, 1)
            hw_P_left = torch.sqrt(energy_m_keV_bucket.clamp(min=-photopeak_offset + 1e-6) + photopeak_offset) * photopeak_scale_left
            d_P_above = energy_m_keV_bucket - Emax
            shrunk = (d_P_above > 0) & (d_P_above <= hw_P_left)

            Emax_t = torch.full((n_bucket, 1), Emax, dtype=torch.float32)
            Emin_t = torch.full((n_bucket, 1), Emin, dtype=torch.float32)
            E1 = torch.where(shrunk, energy_m_keV_bucket - hw_P_left, Emin_t)
            n_res, w_res = self._scale_nodes_exp(E1, Emax_t, *self._nodes_bkg_pool[bkg_nodes[sizes[0]]])
            nodes_out[:] = n_res
            weights_out[:] = w_res
            return nodes_out, weights_out

        energy_m_keV = self._energy_m_keV[event_indices].view(-1, 1)
        phi_rad = self._phi_rad[event_indices].view(-1, 1)

        idx_np = event_indices.numpy()
        sc_coord_sph = self._sc_coord_sph_cache[idx_np]
        lon_ph_rad = torch.as_tensor(asarray(sc_coord_sph.lon.rad, dtype=np.float32))
        lat_ph_rad = torch.as_tensor(asarray(sc_coord_sph.lat.rad, dtype=np.float32))
        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(lon_ph_rad, lat_ph_rad, indices=idx_np)
        phi_geo_rad = phi_geo_rad.view(-1, 1)
        phi_igeo_rad = phi_igeo_rad.view(-1, 1)

        n_peaks = len(seq)
        peak_val         = torch.empty((n_bucket, n_peaks), dtype=torch.float32)
        peak_delta_left  = torch.empty((n_bucket, n_peaks), dtype=torch.float32)
        peak_delta_right = torch.empty((n_bucket, n_peaks), dtype=torch.float32)
        for i, t in enumerate(seq):
            v = self._peak_value(t, energy_m_keV, phi_rad, phi_geo_rad, phi_igeo_rad, width_params).squeeze(1)
            peak_val[:, i] = v
            dl, dr = self._peak_half_width(t, v, width_params)
            peak_delta_left[:, i]  = dl
            peak_delta_right[:, i] = dr

        clamped_left  = (peak_val - peak_delta_left).clone()
        clamped_right = (peak_val + peak_delta_right).clone()
        for i in range(n_peaks - 1):
            mid = (peak_val[:, i] + peak_val[:, i + 1]) / 2.0
            clamped_right[:, i]     = torch.min(clamped_right[:, i], mid)
            clamped_left[:, i + 1]  = torch.max(clamped_left[:, i + 1], mid)
            
        clamped_right = torch.max(clamped_right, clamped_left)  
        clamped_left  = clamped_left.clamp(min=Emin, max=Emax)
        clamped_right = clamped_right.clamp(min=Emin, max=Emax)

        M = n_peaks - 1

        offset = 0
        for i, t in enumerate(seq):
            n_pk = self._peak_node_count(interval_idx, t)
            # EC must lie within [E1,E2] for _scale_nodes_center -- a boundary-admitted
            # peak's TRUE center can legitimately sit outside its own clamped window.
            EC_i = torch.min(torch.max(peak_val[:, i:i + 1], clamped_left[:, i:i + 1]), clamped_right[:, i:i + 1])
            n_res, w_res = self._scale_nodes_center(
                clamped_left[:, i:i + 1], clamped_right[:, i:i + 1], EC_i,
                *self._nodes_peaks_pool[n_pk], self._center_scale_beta[min(t, 2)]
            )
            #n_res, w_res = self._scale_nodes_center(
            #    clamped_left[:, i:i + 1], clamped_right[:, i:i + 1], EC_i,
            #    *self._nodes_peaks_pool[n_pk],
            #    beta=self._center_scale_beta[min(t, 2)],
            #    peak_type=t,
            #)
            nodes_out[:, offset:offset + n_pk] = n_res
            weights_out[:, offset:offset + n_pk] = w_res
            offset += n_pk

            if i < M:
                n_seg = bkg_nodes[sizes[i]]
                n_res, w_res = self._scale_nodes_exp(
                    clamped_right[:, i:i + 1], clamped_left[:, i + 1:i + 2],
                    *self._nodes_bkg_pool[n_seg]
                )
                nodes_out[:, offset:offset + n_seg] = n_res
                weights_out[:, offset:offset + n_seg] = w_res
                offset += n_seg

        if k >= 1:
            if k == 1:
                leading_optional = (seq[0] != self._TYPE_P)
                leading_width  = clamped_left[:, 0] - Emin
                trailing_width = Emax - clamped_right[:, -1]
                leading_present = ((leading_width > presence_eps) if leading_optional
                                    else torch.zeros(n_bucket, dtype=torch.bool)).view(-1, 1)

                n_opt = bkg_nodes[sizes[M]]
                E1 = torch.where(leading_present, torch.full((n_bucket, 1), Emin, dtype=torch.float32),
                                  clamped_right[:, -1:])
                E2 = torch.where(leading_present, clamped_left[:, 0:1],
                                  torch.full((n_bucket, 1), Emax, dtype=torch.float32))
                n_res, w_res = self._scale_nodes_exp(E1, E2, *self._nodes_bkg_pool[n_opt])
                nodes_out[:, offset:offset + n_opt] = n_res
                weights_out[:, offset:offset + n_opt] = w_res
                offset += n_opt
            else:  # k == 2
                n_lead = bkg_nodes[sizes[M]]
                n_res, w_res = self._scale_nodes_exp(Emin, clamped_left[:, 0:1], *self._nodes_bkg_pool[n_lead])
                nodes_out[:, offset:offset + n_lead] = n_res
                weights_out[:, offset:offset + n_lead] = w_res
                offset += n_lead

                n_trail = bkg_nodes[sizes[M + 1]]
                n_res, w_res = self._scale_nodes_exp(clamped_right[:, -1:], Emax, *self._nodes_bkg_pool[n_trail])
                nodes_out[:, offset:offset + n_trail] = n_res
                weights_out[:, offset:offset + n_trail] = w_res
                offset += n_trail

        return nodes_out, weights_out
    
    
    def _update_cache(self):
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        source_coord = self._source.position.sky_coord

        if (self._sc_coord_sph_cache is None) or (source_coord != self._last_convolved_source_skycoord):
            self._sc_coord_sph_cache = self._get_target_in_sc_frame(source_coord, self._sc_ori_unique)[self._inv_idx]

        no_recalculation = ((source_coord == self._last_convolved_source_skycoord)
                             and (self._bucket_irf_cache is not None)
                             and (self._area_cache is not None))

        area_recalculation = ((source_coord != self._last_convolved_source_skycoord)
                               or (self._area_cache is None))

        pdf_recalculation = ((source_coord != self._last_convolved_source_skycoord)
                              or (self._bucket_irf_cache is None))

        if no_recalculation:
            pass
        else:
            active_pool = True
            if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
                active_pool = self._irf.active_pool
                if not active_pool:
                    self._irf.init_compute_pool()

            if source_coord != self._last_convolved_source_skycoord:
                self._bucket_energy_node_cache = None  # geometry changed -- stale, drop it

            if area_recalculation:
                self._compute_area()

            if pdf_recalculation:
                import time
                t0 = time.perf_counter()
                self._compute_density()
                elapsed = time.perf_counter() - t0
                print(f"[TIMING] _compute_density() took {elapsed:.2f} s "
                      f"({self._valid_events} valid events)", flush=True)

            if not active_pool:
                self._irf.shutdown_compute_pool()

            self._last_convolved_source_skycoord = source_coord.copy()

        node_caching = (self.force_energy_node_caching and self._bucket_energy_node_cache is None)
        if node_caching:
            pass
        #    self._compute_nodes() #TODO: Implement
    
    def _compute_line_density(self, line_energy: float, earth_occ_index: np.ndarray) -> torch.Tensor:
        """
        Single fixed-energy contribution (weight=1, no integration), for
        self._energy_range entries that are a plain float rather than [Emin,Emax].
        Every valid event gets the same node energy -- only per-event geometry
        (scatter angle, direction) differs.
        """
        idx_np = self._valid_mask_cache
        n = len(idx_np)
        torch_memory_dtype = torch.float32 if self.reduce_memory else torch.float64
        out = torch.empty(n, dtype=torch_memory_dtype)

        batch_size = max(1, self._cache_batch_size)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sub_idx = idx_np[start:end]

            sc_coord_sph = self._sc_coord_sph_cache[sub_idx]
            lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
            lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
            energies = np.full(end - start, line_energy, dtype=np.float32)

            photons = PhotonListWithDirectionAndEnergyInSCFrame(lon_ph_rad, lat_ph_rad, energies)
            events = EmCDSEventDataInSCFrameFromArrays(
                asarray(self._energy_m_keV[sub_idx], dtype=np.float32),
                asarray(self._lon_scatt[sub_idx], dtype=np.float32),
                asarray(self._lat_scatt[sub_idx], dtype=np.float32),
                asarray(self._phi_rad[sub_idx], dtype=np.float32),
            )
            res = torch.as_tensor(asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float32))
            occ = torch.as_tensor(earth_occ_index[sub_idx])
            live = torch.as_tensor(self._livetime_ratio[sub_idx])
            out[start:end] = (res * occ * live).to(torch_memory_dtype)

        return out
    
    def _compute_density_helper(self, interval_idx: int, bucket_id: int, event_indices: torch.Tensor,
                                source_coord, Emin: float, Emax: float,
                                earth_occ_index: Optional[np.ndarray] = None,
                                live_ratio: Optional[np.ndarray] = None
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Queries the IRF for one bucket's events (or a chunk of one), folds in
        occultation/livetime/quadrature weights.
        """
        idx_np = event_indices.numpy()

        nodes, weights = self._fill_bucket_nodes(interval_idx, bucket_id, event_indices, Emin, Emax)
        n_sub, n_nodes = nodes.shape

        if earth_occ_index is None:
            earth_occ_sub = self._earth_occ(source_coord, self._sc_ori_unique)[self._inv_idx[idx_np]]
        else:
            earth_occ_sub = earth_occ_index[idx_np]
        live_sub = (self._livetime_ratio[idx_np] if live_ratio is None else live_ratio[idx_np])

        e_sl = asarray(self._energy_m_keV[event_indices], dtype=np.float32)
        p_sl = asarray(self._phi_rad[event_indices], dtype=np.float32)

        sc_coord_sph = self._sc_coord_sph_cache[idx_np]
        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

        photons = PhotonListWithDirectionAndEnergyInSCFrame(
            np.repeat(lon_ph_rad, n_nodes),
            np.repeat(lat_ph_rad, n_nodes),
            np.asarray(nodes).ravel(),
        )
        events = EmCDSEventDataInSCFrameFromArrays(
            np.repeat(e_sl, n_nodes),
            np.repeat(asarray(self._lon_scatt[event_indices], dtype=np.float32), n_nodes),
            np.repeat(asarray(self._lat_scatt[event_indices], dtype=np.float32), n_nodes),
            np.repeat(p_sl, n_nodes),
        )

        res_block = torch.as_tensor(
            asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float32)
        ).view(n_sub, n_nodes)

        occ = torch.as_tensor(earth_occ_sub).view(-1, 1)
        live = torch.as_tensor(live_sub).view(-1, 1)
        res_block = res_block * occ * live * weights

        return res_block, nodes
    
    def _compute_density(self):
        coord = self._source.position.sky_coord
        torch_memory_dtype = torch.float32 if self.reduce_memory else torch.float64
        np_memory_dtype = np.float32 if self.reduce_memory else np.float64

        earth_occ_index = self._earth_occ(coord, self._sc_ori_unique)[self._inv_idx]
        self._valid_mask_cache = np.where((earth_occ_index > 0) & (self._livetime_ratio > 0))[0]
        self._valid_events = int(len(self._valid_mask_cache))
        self._valid_mask_tensor = torch.as_tensor(self._valid_mask_cache, dtype=torch.long)

        self._bucket_event_indices = []
        self._bucket_irf_cache = []
        self._line_irf_cache = []
        self._bucket_energy_node_cache = [] if self.force_energy_node_caching else None

        if self._valid_events == 0:
            empty_line = torch.empty(0, dtype=torch_memory_dtype)

            for item in self._energy_range:
                is_interval = isinstance(item, list)
                self._bucket_event_indices.append({} if is_interval else None)
                self._bucket_irf_cache.append({} if is_interval else None)
                self._line_irf_cache.append(None if is_interval else empty_line)
                if self._bucket_energy_node_cache is not None:
                    self._bucket_energy_node_cache.append({} if is_interval else None)
            return

        #sc_coord_sph_valid = self._sc_coord_sph_cache[self._valid_mask_cache]
        #lon_ph_rad = torch.as_tensor(asarray(sc_coord_sph_valid.lon.rad, dtype=np.float32))
        #lat_ph_rad = torch.as_tensor(asarray(sc_coord_sph_valid.lat.rad, dtype=np.float32))
        #phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(lon_ph_rad, lat_ph_rad, indices=self._valid_mask_cache)

        #energy_valid = self._energy_m_keV[self._valid_mask_tensor]
        #phi_valid = self._phi_rad[self._valid_mask_tensor]

        interval_idx = 0
        for item in self._energy_range:
            if not isinstance(item, list):
                self._bucket_event_indices.append(None)
                self._bucket_irf_cache.append(None)
                if self._bucket_energy_node_cache is not None:
                    self._bucket_energy_node_cache.append(None)
                self._line_irf_cache.append(self._compute_line_density(float(item), earth_occ_index))
                continue

            self._line_irf_cache.append(None)
            Emin, Emax = item
            
            #bucket_id_local = self._classify_interval(
            #    energy_valid, phi_valid, phi_geo_rad, phi_igeo_rad,
            #    interval_idx, float(Emin), float(Emax),
            #)
            bucket_id_chunks = []
            for start in range(0, self._valid_events, self._cache_batch_size):
                end = min(start + self._cache_batch_size, self._valid_events)
                sub_valid_idx = self._valid_mask_cache[start:end]

                sc_coord_sph_sub = self._sc_coord_sph_cache[sub_valid_idx]
                lon_sub = torch.as_tensor(asarray(sc_coord_sph_sub.lon.rad, dtype=np.float32))
                lat_sub = torch.as_tensor(asarray(sc_coord_sph_sub.lat.rad, dtype=np.float32))
                phi_geo_sub, phi_igeo_sub = self._get_CDS_coordinates(lon_sub, lat_sub, indices=sub_valid_idx)

                chunk_bucket_id = self._classify_interval(
                    self._energy_m_keV[sub_valid_idx], self._phi_rad[sub_valid_idx],
                    phi_geo_sub, phi_igeo_sub,
                    interval_idx, float(Emin), float(Emax),
                )
                bucket_id_chunks.append(chunk_bucket_id)

            bucket_id_local = torch.cat(bucket_id_chunks)
            n_buckets = len(self._bucket_tables[interval_idx])
            grouped_local = self._group_by_bucket(bucket_id_local, n_buckets)
            grouped_original = {bid: self._valid_mask_tensor[local_idx] for bid, local_idx in grouped_local.items()}
            self._bucket_event_indices.append(grouped_original)

            irf_cache_interval = {}
            node_cache_interval = {} if self._bucket_energy_node_cache is not None else None

            for bid, indices in tqdm(grouped_original.items(),
                                      disable=(not self.show_progress),
                                      desc=f"Caching the response (interval {interval_idx})",
                                      smoothing=0.2, leave=False):
                n_nodes_bucket = self._bucket_tables[interval_idx][bid]['n_nodes']
                if n_nodes_bucket == 0:
                    continue
                
                n_bucket_events = indices.shape[0]
                batch_size = max(1, self._cache_batch_size // n_nodes_bucket)

                irf_bucket = torch.empty((n_bucket_events, n_nodes_bucket), dtype=torch_memory_dtype)
                node_bucket = (np.empty((n_bucket_events, n_nodes_bucket), dtype=np_memory_dtype)
                               if node_cache_interval is not None else None)

                for start in range(0, n_bucket_events, batch_size):
                    end = min(start + batch_size, n_bucket_events)
                    res_block, nodes = self._compute_density_helper(
                        interval_idx, bid, indices[start:end], coord, float(Emin), float(Emax),
                        earth_occ_index=earth_occ_index, live_ratio=self._livetime_ratio,
                    )
                    irf_bucket[start:end] = res_block.to(torch_memory_dtype)
                    if node_bucket is not None:
                        node_bucket[start:end] = nodes.numpy().astype(np_memory_dtype)

                irf_cache_interval[bid] = irf_bucket
                if node_cache_interval is not None:
                    node_cache_interval[bid] = node_bucket

            self._bucket_irf_cache.append(irf_cache_interval)
            if node_cache_interval is not None:
                self._bucket_energy_node_cache.append(node_cache_interval)

            interval_idx += 1
    
    def _compute_area(self):
        coord = self._source.position.sky_coord
        n_energy = self.total_expectation_integration_nodes
        
        e_n, e_w = [], []

        for x in self._energy_range:
            if isinstance(x, (float, int)):
                e_n.append([float(x)])
                e_w.append([1.0])
            else:
                E1, E2 = torch.tensor(x[0]), torch.tensor(x[1])
                n_nodes = self._total_expectation_integration_nodes(self._total_expectation_resolution, E2 - E1)

                n_np, w_np = np.polynomial.legendre.leggauss(n_nodes)

                n_torch = torch.from_numpy(n_np)
                w_torch = torch.from_numpy(w_np)

                n, w = self._scale_nodes_log(E1, E2, n_torch, w_torch)

                e_n.append(n.numpy())
                e_w.append(w.numpy())

        self._area_energy_node_cache = np.concatenate(e_n).astype(np.float64)
        e_n = self._area_energy_node_cache.astype(np.float32)
        e_w = np.concatenate(e_w).astype(np.float32)

        # Midpoint
        sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_center)
        earth_occ_index = self._earth_occ(coord, self._sc_ori_center)
        
        combined_time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(np.float32) * earth_occ_index

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

        n_time = len(lon_ph_rad)
        batch_size_time = self._cache_batch_size // n_energy

        total_area = np.zeros(n_energy, dtype=np.float64)
        
        max_batch_total = n_energy * min(batch_size_time, n_time)
        batch_lons_buffer = np.empty(max_batch_total, dtype=np.float32)
        batch_lats_buffer = np.empty(max_batch_total, dtype=np.float32)
        batch_energies_buffer = np.empty(max_batch_total, dtype=np.float32)
        
        for i in tqdm(range(0, n_time, batch_size_time), 
                      disable=(not self.show_progress),
                      desc="Caching the effective area", 
                      smoothing=0.2, 
                      leave=False):
            start = i
            end = min(i + batch_size_time, n_time)
            current_n_time = end - start
            current_total = current_n_time * n_energy

            batch_lons_buffer[:current_total].reshape(current_n_time, n_energy)[:] = lon_ph_rad[start:end, np.newaxis]
            batch_lats_buffer[:current_total].reshape(current_n_time, n_energy)[:] = lat_ph_rad[start:end, np.newaxis]
            batch_energies_buffer[:current_total].reshape(current_n_time, n_energy)[:] = e_n
            
            photons = PhotonListWithDirectionAndEnergyInSCFrame(
                batch_lons_buffer[:current_total],
                batch_lats_buffer[:current_total],
                batch_energies_buffer[:current_total]
                )
            
            eff_areas_flat = asarray(self._irf.effective_area_cm2(photons), dtype=np.float32)
            eff_areas_grid = eff_areas_flat.reshape(current_n_time, n_energy)

            total_area += np.einsum('ij,i,j->j', 
                                    eff_areas_grid, 
                                    combined_time_weights[start:end], 
                                    e_w.ravel())

        self._area_cache = total_area
    
    
    def _integrate_single_event_density(self, event_idx: int, occ_val: float, live_val: float, 
                                        lon_val: float, lat_val: float, 
                                        relerr: float, abserr: float, maxEval: int) -> tuple[float, bool]: 
        from cubature import cubature
        
        if np.isclose(0.0, occ_val * live_val):
            return 0.0, True

        e_meas = self._energy_m_keV[event_idx].item()
        lon_sc = self._lon_scatt[event_idx].item()
        lat_sc = self._lat_scatt[event_idx].item()
        phi_m = self._phi_rad[event_idx].item()

        def density_integrand(Ei):
            energies = Ei[:, 0]
            num_eval = len(energies)
            
            lon_src = np.full(num_eval, lon_val, dtype=np.float32)
            lat_src = np.full(num_eval, lat_val, dtype=np.float32)
            e_meas_arr = np.full(num_eval, e_meas, dtype=np.float32)
            lon_sc_arr = np.full(num_eval, lon_sc, dtype=np.float32)
            lat_sc_arr = np.full(num_eval, lat_sc, dtype=np.float32)
            phi_m_arr = np.full(num_eval, phi_m, dtype=np.float32)

            photons = PhotonListWithDirectionAndEnergyInSCFrame(lon_src, lat_src, energies.astype(np.float32))
            events = EmCDSEventDataInSCFrameFromArrays(e_meas_arr, lon_sc_arr, lat_sc_arr, phi_m_arr)

            diff_area = asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float64)
            flux = asarray(self._source(energies), dtype=np.float64)
            return diff_area * occ_val * live_val * flux

        total_res_val = 0.0
        total_err = 0.0
        
        for x in self._energy_range:
            if isinstance(x, list):
                res, err = cubature(
                    density_integrand, ndim=1, fdim=1,
                    xmin=[x[0]], xmax=[x[1]],
                    vectorized=True, relerr=relerr, abserr=abserr, maxEval=maxEval
                )
                total_res_val += res[0]
                total_err += err[0]
            else:
                Ei = np.array([[float(x)]], dtype=np.float64)
                val = density_integrand(Ei)
                total_res_val += val[0]
        
        allowed_err = max(abserr, relerr * abs(total_res_val))
        return total_res_val, bool(total_err <= allowed_err)

    def _integrate_total_counts(self, coord, relerr: float, abserr: float, maxEval: int) -> tuple[float, bool]:
        from cubature import cubature
        
        sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_center)
        lon_ph_rad_center = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad_center = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        earth_occ_center = self._earth_occ(coord, self._sc_ori_center)
        combined_time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(np.float32) * earth_occ_center

        def total_counts_integrand(Ei):
            energies = Ei[:, 0]
            num_eval = len(energies)
            total_counts_for_energies = np.zeros(num_eval, dtype=np.float64)
            
            for i_e, E in enumerate(energies):
                batch_energies = np.full(len(lon_ph_rad_center), E, dtype=np.float32)
                photons = PhotonListWithDirectionAndEnergyInSCFrame(lon_ph_rad_center, lat_ph_rad_center, batch_energies)
                eff_areas = asarray(self._irf.effective_area_cm2(photons), dtype=np.float64)
                
                total_area_at_E = np.sum(eff_areas * combined_time_weights)
                flux_at_E = self._source(np.array([E]))[0]
                total_counts_for_energies[i_e] = total_area_at_E * flux_at_E
                
            return total_counts_for_energies

        high_prec_total_counts = 0.0
        total_err_total = 0.0

        for x in self._energy_range:
            if isinstance(x, list):
                res_total, err_total = cubature(
                    total_counts_integrand, ndim=1, fdim=1,
                    xmin=[x[0]], xmax=[x[1]],
                    vectorized=True, relerr=relerr, abserr=abserr, maxEval=maxEval
                )
                high_prec_total_counts += float(res_total[0])
                total_err_total += float(err_total[0])
            else:
                Ei = np.array([[float(x)]], dtype=np.float64)
                val_total = total_counts_integrand(Ei)
                high_prec_total_counts += float(val_total[0])

        allowed_err_total = max(abserr, relerr * abs(high_prec_total_counts))
        return high_prec_total_counts, bool(total_err_total <= allowed_err_total)

    def _compute_density_for_indices(self, indices: np.ndarray, coord) -> np.ndarray:
        """
        Self-contained density computation for an arbitrary small set of original
        event indices.
        """
        if (self._sc_coord_sph_cache is None) or (coord != self._last_convolved_source_skycoord):
            self._sc_coord_sph_cache = self._get_target_in_sc_frame(coord, self._sc_ori_unique)[self._inv_idx]

        n = len(indices)
        indices_t = torch.as_tensor(indices, dtype=torch.long)
        earth_occ_full = self._earth_occ(coord, self._sc_ori_unique)[self._inv_idx]  # (n_events,)

        sc_coord_sph_sub = self._sc_coord_sph_cache[indices]
        lon_ph_rad = torch.as_tensor(asarray(sc_coord_sph_sub.lon.rad, dtype=np.float32))
        lat_ph_rad = torch.as_tensor(asarray(sc_coord_sph_sub.lat.rad, dtype=np.float32))

        exp_density = torch.zeros(n, dtype=torch.float64)
        total_nodes_evaluated = 0
        peak_eval_counts = {0: 0, 1: 0, 2: 0, 3: 0}   # Anz. Event-Durchläufe pro Peak-Klasse
        peak_node_counts = {0: 0, 1: 0, 2: 0, 3: 0}   # Anz. Knoten gesamt pro Peak-Klasse # TODO: DEBUG

        interval_idx = 0
        for item in self._energy_range:
            if not isinstance(item, list):
                energies = np.full(n, float(item), dtype=np.float32)
                photons = PhotonListWithDirectionAndEnergyInSCFrame(
                    asarray(lon_ph_rad, dtype=np.float32), asarray(lat_ph_rad, dtype=np.float32), energies
                )
                events = EmCDSEventDataInSCFrameFromArrays(
                    asarray(self._energy_m_keV[indices_t], dtype=np.float32),
                    asarray(self._lon_scatt[indices_t], dtype=np.float32),
                    asarray(self._lat_scatt[indices_t], dtype=np.float32),
                    asarray(self._phi_rad[indices_t], dtype=np.float32),
                )
                res = torch.as_tensor(asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float64))
                occ = torch.as_tensor(earth_occ_full[indices], dtype=torch.float64)
                live = torch.as_tensor(self._livetime_ratio[indices], dtype=torch.float64)
                flux_line = float(self._source(np.array([float(item)], dtype=np.float64))[0])
                exp_density += res * occ * live * flux_line
                continue

            Emin, Emax = item
            phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(lon_ph_rad, lat_ph_rad, indices=indices)

            bucket_id_local = self._classify_interval(
                self._energy_m_keV[indices_t], self._phi_rad[indices_t],
                phi_geo_rad, phi_igeo_rad, interval_idx, float(Emin), float(Emax),
            )
            n_buckets = len(self._bucket_tables[interval_idx])
            grouped_local = self._group_by_bucket(bucket_id_local, n_buckets)

            for bid, local_idx in grouped_local.items():
                n_nodes_bucket = self._bucket_tables[interval_idx][bid]['n_nodes']
                n_peaks = len(self._bucket_tables[interval_idx][bid]['seq']) # TODO: DEBUG
                if n_nodes_bucket == 0:
                    continue

                sub_original_indices = indices_t[local_idx]
                res_block, nodes = self._compute_density_helper(
                    interval_idx, bid, sub_original_indices, coord, float(Emin), float(Emax),
                    earth_occ_index=earth_occ_full, live_ratio=self._livetime_ratio,
                )
                flux = torch.as_tensor(self._source(nodes.numpy()), dtype=torch.float64)
                contribution = torch.linalg.vecdot(res_block.to(torch.float64), flux, dim=1)
                exp_density.index_add_(0, local_idx, contribution)
                
                n_sub = len(local_idx)
                total_nodes_evaluated += n_sub * n_nodes_bucket
                if n_peaks in peak_eval_counts:
                    peak_eval_counts[n_peaks] += n_sub
                    peak_node_counts[n_peaks] += n_sub * n_nodes_bucket # TODO: DEBUG

            interval_idx += 1
        
        total_interval_evals = sum(peak_eval_counts.values())
        avg_total_nodes = total_nodes_evaluated / n if n > 0 else 0.0

        print("\n" + "=" * 65, flush=True)
        print(f"[DEBUG PROFILING] Total Events: {n} | Total Evaluated Nodes: {total_nodes_evaluated}")
        print(f"[DEBUG PROFILING] Average Nodes per Event (overall): {avg_total_nodes:.2f}")
        print("-" * 65, flush=True)
        print(f"{'Peaks':<8} | {'Evaluations':<12} | {'Share (%)':<10} | {'Avg Nodes/Event':<16}")
        print("-" * 65, flush=True)

        for k in range(4):
            evals = peak_eval_counts[k]
            nodes = peak_node_counts[k]
            pct = (evals / total_interval_evals * 100.0) if total_interval_evals > 0 else 0.0
            avg_nodes_k = (nodes / evals) if evals > 0 else 0.0
            print(f"{k:<8} | {evals:<12} | {pct:<9.1f}% | {avg_nodes_k:<16.2f}", flush=True)

        print("=" * 65 + "\n", flush=True) # TODO: DEBUG

        result = np.asarray(exp_density, dtype=np.float64)
        if self._offset is not None:
            result += self._offset

        return result

    def validate_integration(self, 
                             n_events: Optional[int] = None,
                             indices: Optional[Union[int, Sequence[int], np.ndarray]] = None,
                             check_total_counts: bool = True,
                             relerr: float = 4e-4, 
                             abserr: float = 1e-30, 
                             maxEval: int = 1000000,
                             save_path: Optional[Union[str, Path]] = None) -> dict:
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        if indices is None and n_events is None:
            raise ValueError("Either 'n_events' or 'indices' must be specified.")

        print("Running validation", flush=True)
        pool_was_active = True
        if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
            pool_was_active = self._irf.active_pool
            if not pool_was_active:
                self._irf.init_compute_pool()
        print("Pool initialized", flush=True)

        try:
            coord = self._source.position.sky_coord

            sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_unique)[self._inv_idx]
            earth_occ_index = self._earth_occ(coord, self._sc_ori_unique)[self._inv_idx]
            self._valid_mask_cache = np.where((earth_occ_index > 0) & (self._livetime_ratio > 0))[0]
            self._valid_events = int(len(self._valid_mask_cache))
            print("Computed valid events", flush=True)

            if indices is not None:
                sampled_indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
            else:
                sampled_indices = np.random.choice(
                    self._valid_mask_cache, 
                    size=min(n_events, self._valid_events), 
                    replace=False
                )

            print("Determined sampled indices", flush=True)

            sc_coord_sph_sampled = sc_coord_sph[sampled_indices]
            earth_occ_sampled = earth_occ_index[sampled_indices]
            livetime_ratio_sampled = self._livetime_ratio[sampled_indices]

            print("Determined sampled arrays", flush=True)

            high_prec_densities = []
            converged_mask = []

            for idx, event_idx in enumerate(tqdm(sampled_indices, desc="Computing high-precision densities", disable=not self.show_progress)):
                res_val, is_converged = self._integrate_single_event_density(
                    event_idx, earth_occ_sampled[idx], livetime_ratio_sampled[idx],
                    sc_coord_sph_sampled[idx].lon.rad, sc_coord_sph_sampled[idx].lat.rad,
                    relerr, abserr, maxEval
                )
                if self._offset is not None:
                    res_val += self._offset
                high_prec_densities.append(res_val)
                converged_mask.append(is_converged)

            print("Finished computing high-precision densities", flush=True)

            high_prec_densities = np.array(high_prec_densities, dtype=np.float64)
            converged_mask = np.array(converged_mask, dtype=bool)

            if sum(converged_mask) / len(converged_mask) < 0.9:
                raise RuntimeError("Less than 90% of high-precision densities converged. Try increasing the maximum number of evaluations or the acceptable error.")

            # --- Total counts computation (Optional) ---
            if check_total_counts:
                print("Determine high-precision total counts", flush=True)
                high_prec_total_counts, total_counts_converged = self._integrate_total_counts(coord, relerr, abserr, maxEval)
                print("Finished computing high-precision total counts", flush=True)

                if not total_counts_converged:
                    raise RuntimeError("High-precision total counts integration did not converge. Try increasing the maximum number of evaluations or the acceptable error.")

                if (self._area_cache is None) or (coord != self._last_convolved_source_skycoord):
                    self._compute_area()
                flux_area = self._source(self._area_energy_node_cache)
                opt_total_counts = float(np.sum(self._area_cache * flux_area, dtype=float))
                rel_deviation_total = (opt_total_counts / high_prec_total_counts) - 1.0

                total_counts_dict = {
                    "optimized": opt_total_counts,
                    "high_precision": high_prec_total_counts,
                    "relative_deviation": rel_deviation_total,
                }
            else:
                total_counts_dict = None

            opt_densities = self._compute_density_for_indices(sampled_indices, coord)
            print("Finished computing optimized densities", flush=True)

            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = (opt_densities / high_prec_densities) - 1.0
                rel_errors = np.nan_to_num(rel_errors, nan=0.0, posinf=0.0, neginf=0.0)

            converged_rel_errors = rel_errors[converged_mask]

            if len(converged_rel_errors) > 0:
                mean_err = float(np.mean(converged_rel_errors))
                median_err = float(np.median(converged_rel_errors))
                std_err = float(np.std(converged_rel_errors, ddof=1)) if len(converged_rel_errors) > 1 else 0.0
            else:
                mean_err = median_err = std_err = float('nan')

            results = {
                "total_counts": total_counts_dict,
                "density_errors": {
                    "mean": mean_err,
                    "median": median_err,
                    "std": std_err,
                    "raw_relative_errors": rel_errors,
                    "sampled_indices": sampled_indices,
                    "converged_mask": converged_mask,
                    "num_converged": int(np.sum(converged_mask)),
                    "total_sampled": len(sampled_indices)
                }
            }

            if save_path is not None:
                serializable_results = {
                    "total_counts": results["total_counts"],
                    "density_errors": {
                        "mean": results["density_errors"]["mean"],
                        "median": results["density_errors"]["median"],
                        "std": results["density_errors"]["std"],
                        "num_converged": results["density_errors"]["num_converged"],
                        "total_sampled": results["density_errors"]["total_sampled"],
                        "raw_relative_errors": results["density_errors"]["raw_relative_errors"].tolist(),
                        "sampled_indices": results["density_errors"]["sampled_indices"].tolist(),
                        "converged_mask": results["density_errors"]["converged_mask"].tolist()
                    }
                }
                with open(str(save_path), 'w') as f:
                    json.dump(serializable_results, f, indent=4)

            return results

        finally:
            if not pool_was_active and isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
                self._irf.shutdown_compute_pool()


    @staticmethod
    def report_and_plot_validation(validation_results: Union[dict, str, Path], 
                                   save_path: Optional[Union[str, Path]] = None):
        if isinstance(validation_results, (str, Path)):
            if not os.path.exists(str(validation_results)):
                raise FileNotFoundError(f"Validation file {str(validation_results)} not found.")
            with open(str(validation_results), 'r') as f:
                validation_results = json.load(f)

        tc = validation_results["total_counts"]
        de = validation_results["density_errors"]

        print("=" * 55)
        print("         INTEGRATION VALIDATION REPORT")
        print("=" * 55)
        if tc is not None:
            print(f"Total Expected Counts (Optimized):  {tc['optimized']:.2f}")
            print(f"Total Expected Counts (Reference): {tc['high_precision']:.2f}")
            print(f"Relative Deviation:   {tc['relative_deviation']*100:.1e}%")
            print("-" * 55)
        print("Event Density Relative Errors:")
        print(f"  Convergence:  {de['num_converged']/de['total_sampled']*100:.2f}% of all sampled events converged")
        print(f"  Mean Error:   {de['mean']*100:.1e}%")
        print(f"  Median Error: {de['median']*100:.1e}%")
        print(f"  Std Dev:      {de['std']*100:.1e}%")
        print("=" * 55)

        errors = np.array(de["raw_relative_errors"]) * 100

        if "converged_mask" in de:
            converged_mask = np.array(de["converged_mask"], dtype=bool)
        else:
            converged_mask = np.ones(len(errors), dtype=bool)

        converged_errors = errors[converged_mask]

        if len(converged_errors) == 0:
            raise RuntimeError("Zero events converged successfully.")

        fig, ax = plt.subplots(figsize=(9, 6))
        
        p1, p99 = np.percentile(converged_errors, [1, 99])
        plot_range = (p1, p99) if p1 < p99 else (converged_errors.min(), converged_errors.max())
        
        ax.hist(converged_errors, bins='fd', range=plot_range, alpha=0.75, color='royalblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        ax.axvline(de['median']*100, color='darkorange', linestyle='-', 
                   label=f"Median ({de['median']*100:.1e}%)")
        ax.set_xlabel("Relative Error (%)", fontsize=11)

        ax.set_title("Distribution of Relative Errors in Expectation Densities", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Events", fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
            
        plt.show()
    
    def expected_counts(self) -> float:
        """
        Return the total expected counts.
        """
        self._update_cache()
        source_dict = self._source.to_dict()
        
        if (source_dict != self._last_convolved_source_dict_number) or (self._exp_events is None):
            area = self._area_cache
            flux = self._source(self._area_energy_node_cache)
            self._exp_events = np.sum(area * flux, dtype=float)
            
        self._last_convolved_source_dict_number = source_dict
        return self._exp_events
    
    def _compute_expectation_density(self):
        """
        Combines the cached IRF response (self._bucket_irf_cache, response * occ *
        livetime * quadrature weights, already baked in during _compute_density)
        with the current source flux, per bucket, and accumulates into
        self._exp_density in original event order.

        Two paths per bucket, chosen automatically:
          - node cache present  -> flux evaluated at the stored node positions
          - node cache absent   -> nodes regenerated on the fly via _fill_bucket_nodes
        """
        self._exp_density = torch.zeros(self._n_events, dtype=torch.float64)

        if self._valid_events == 0:
            return

        interval_idx = 0
        for item_idx, item in enumerate(self._energy_range):

            # ---- single fixed-energy line: no integration, one node per event ----
            if not isinstance(item, list):
                line_cache = self._line_irf_cache[item_idx]  # (valid_events,), response*occ*live already applied
                flux_line = float(self._source(np.array([float(item)], dtype=np.float64))[0])
                contribution = line_cache.to(torch.float64) * flux_line
                self._exp_density.index_add_(0, self._valid_mask_tensor, contribution)
                continue

            # ---- integrated interval: per-bucket dot product ----
            Emin, Emax = item
            bucket_indices      = self._bucket_event_indices[item_idx]
            irf_cache_interval  = self._bucket_irf_cache[item_idx]
            node_cache_interval = (self._bucket_energy_node_cache[item_idx]
                                   if self._bucket_energy_node_cache is not None else None)

            for bid, indices in bucket_indices.items():
                n_nodes_bucket = self._bucket_tables[interval_idx][bid]['n_nodes']
                if n_nodes_bucket == 0:
                    continue  # empty bucket (interval entirely below photopeak reach) -- zero contribution

                cache = irf_cache_interval[bid]  # (n_bucket, n_nodes_bucket)
                n_bucket_events = indices.shape[0]
                batch_size = max(1, self._integration_batch_size // n_nodes_bucket)

                for start in range(0, n_bucket_events, batch_size):
                    end = min(start + batch_size, n_bucket_events)
                    sub_indices = indices[start:end]
                    sub_cache = cache[start:end]

                    if node_cache_interval is not None:
                        nodes_np = node_cache_interval[bid][start:end]
                    else:
                        nodes_t, _ = self._fill_bucket_nodes(interval_idx, bid, sub_indices, float(Emin), float(Emax))
                        nodes_np = nodes_t.numpy()

                    flux_np = self._source(nodes_np)
                    flux = torch.as_tensor(flux_np, dtype=torch.float64)

                    contribution = torch.linalg.vecdot(sub_cache.to(torch.float64), flux, dim=1)
                    self._exp_density.index_add_(0, sub_indices, contribution)

            interval_idx += 1
    
    def expectation_density(self) -> torch.Tensor:
        """
        Return the expectation density for each event.
        """
        self._update_cache()
        source_dict = self._source.to_dict()

        if (source_dict != self._last_convolved_source_dict_density) or (self._exp_density is None):
            self._compute_expectation_density()
            self._last_convolved_source_dict_density = source_dict

        if self._offset is not None:
            return self._exp_density + self._offset
        return self._exp_density