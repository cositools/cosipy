import copy
import os
import json
from typing import Optional, Iterable, Type, Tuple, List, Union
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numpy as np
import h5py
from astromodels import ExtendedSource
from astropy.coordinates import CartesianRepresentation
from executing import Source
from scoords import SpacecraftFrame
import healpy as hp 

from cosipy import SpacecraftHistory
from cosipy.interfaces.source_response_interface import (
    CachedUnbinnedThreeMLSourceResponseInterface,
)
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces import EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import (
    FarFieldSpectralInstrumentResponseFunctionInterface,
)
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.util.iterables import asarray

from astropy import units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord
from astropy.time import Time

import logging

logger = logging.getLogger(__name__)

import torch
from cosipy.response.ml.nf_instrument_response_function import (
    UnpolarizedNFFarFieldInstrumentResponseFunction,
)
from cosipy.response.ml.NFNormalizationBase import IEnergyList


PeakNodeList = Union[Tuple[int, int], List[Tuple[int, int]]]
DensityNodeList = Union[int, List[int], Tuple[int, ...]]


class UnbinnedThreeMLExtendedSourceResponseIRFAdaptive(
    CachedUnbinnedThreeMLSourceResponseInterface
):

    def __init__(
        self,
        data: TimeTagEmCDSEventDataInSCFrameInterface,
        irf: FarFieldSpectralInstrumentResponseFunctionInterface,
        sc_history: SpacecraftHistory,
        show_progress: bool = True,
        force_energy_node_caching: bool = False,
        reduce_memory: bool = True,
        nside = 64,
        mask = None
    ):
        """
        Will fold the IRF with the extended source spectrum by evaluating the IRF at Ei positions adaptively chosen based on characteristic IRF features
        Note that this assumes a smooth flux spectrum

        All IRF queries are cached and can be saved to / loaded from a file


        Parameters
        ----------
        nside : int
            nside value used for evaluating the all-sky. Default is nside = 64.
        mask : array-like of booleans, optional
            array of size npix which tell which pixel to mask.
        """

        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._irf = irf
        self._sc_ori = sc_history
        self.show_progress = show_progress
        self.force_energy_node_caching = force_energy_node_caching
        
        #extended source map inputs
        self._nside = nside
        self._mask = mask
        self._pixels = None #list of pixels to use
        self._npix = hp.nside2npix(nside)
        self._pixelsolidang = hp.nside2pixarea(nside) #solid ang per pixel
        self._lon = None #list of lon for each pixel
        self._lat = None #list of lat for each pixel

        # Default parameters for irf energy node placement
        self._density_integration_nodes = [
            60,
        ]
        self._total_expectation_resolution = 18.0
        self._peak_nodes = [
            [18, 12],
        ]
        self._peak_widths: Tuple[float, float] = (0.04, 0.1)
        self._energy_range = [
            [100.0, 10_000.0],
        ]
        self._n_intervals = 1
        self._cache_batch_size = 1_000_000
        self._integration_batch_size = 1_000_000
        self._offset: Optional[float] = 1e-12

        # Placeholder for node pool - stored as Tensors
        self._width_tensor: Optional[torch.Tensor] = None
        self._nodes_primary: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_secondary: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        self._nodes_bkg_0: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_bkg_1: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_bkg_2: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = (
            None
        )
        self._nodes_bkg_3: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = (
            None
        )

        # Checks to avoid unecessary recomputations
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache: list[np.ndarray] | None = None
        self._source_coord_cache = None
        
        # Cached values
        self._irf_cache: list[torch.Tensor] | None = None  # cm^2/rad/sr
        self._irf_energy_node_cache: list[np.ndarray] | None = None # (Optional, only if full batch)
        
        self._area_cache: Optional[np.ndarray] = None  # cm^2*s*keV
        self._area_energy_node_cache: Optional[np.ndarray] = None
        self._exp_events: Optional[float] = None
        self._exp_density: Optional[torch.Tensor] = None
        self._valid_mask_cache: list[np.ndarray] | None = None
        self._valid_events: Optional[int] = None

        # Precomputed spacecraft history - Midpoint
        self._mid_times = (
            self._sc_ori.obstime[:-1]
            + (self._sc_ori.obstime[1:] - self._sc_ori.obstime[:-1]) / 2
        )
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
        self._unique_unix, self._inv_idx = np.unique(
            data_times.utc.unix, return_inverse=True
        )
        unique_times_obj = Time(self._unique_unix, format="unix", scale="utc")
        self._sc_ori_unique = self._sc_ori.interp(unique_times_obj)

        interval_ratios = self._sc_ori.livetime.to_value(
            u.s
        ) / self._sc_ori.intervals_duration.to_value(u.s)
        bin_indices = (
            np.searchsorted(
                self._sc_ori.obstime.utc.unix, self._unique_unix, side="right"
            )
            - 1
        )
        bin_indices = np.clip(bin_indices, 0, len(self._sc_ori.livetime) - 1)
        unique_ratio = interval_ratios[bin_indices]
        self._livetime_ratio = unique_ratio[self._inv_idx].astype(np.float32)

        self._energy_m_keV = torch.as_tensor(
            asarray(self._data.energy_keV, dtype=np.float32)
        )
        self._phi_rad = torch.as_tensor(
            asarray(self._data.scattering_angle_rad, dtype=np.float32)
        )

        self._lon_scatt = torch.as_tensor(
            asarray(self._data.scattered_lon_rad_sc, dtype=np.float32)
        )
        self._lat_scatt = torch.as_tensor(
            asarray(self._data.scattered_lat_rad_sc, dtype=np.float32)
        )
        self._cos_lat_scatt = torch.cos(self._lat_scatt)
        self._sin_lat_scatt = torch.sin(self._lat_scatt)
        self._cos_lon_scatt = torch.cos(self._lon_scatt)
        self._sin_lon_scatt = torch.sin(self._lon_scatt)

        # Also runs _check_memory_savings
        self.reduce_memory = reduce_memory

    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface

    @property
    def force_energy_node_caching(self) -> bool:
        return self._force_energy_node_caching

    @force_energy_node_caching.setter
    def force_energy_node_caching(self, val):
        if not isinstance(val, bool):
            raise ValueError("force_energy_node_caching must be a boolean")
        self._force_energy_node_caching = val

    @property
    def density_integration_nodes(self):
        return self._density_integration_nodes

    @density_integration_nodes.setter
    def density_integration_nodes(self, val):
        self.set_integration_parameters(density_integration_nodes=val)

    @property
    def total_expectation_resolution(self) -> float:
        return self._total_expectation_resolution

    @total_expectation_resolution.setter
    def total_expectation_resolution(self, val):
        self.set_integration_parameters(total_expectation_resolution=val)

    @property
    def peak_nodes(self):
        return [tuple(x) for x in self._peak_nodes]

    @peak_nodes.setter
    def peak_nodes(self, val):
        self.set_integration_parameters(peak_nodes=val)

    @property
    def peak_widths(self) -> Tuple[float, float]:
        return self._peak_widths

    @peak_widths.setter
    def peak_widths(self, val):
        self.set_integration_parameters(peak_widths=val)

    @property
    def energy_range(self):
        return [tuple(x) if isinstance(x, list) else x for x in self._energy_range]

    @energy_range.setter
    def energy_range(self, val):
        self.set_integration_parameters(energy_range=val)

    @property
    def n_intervals(self) -> int:
        return self._n_intervals

    @property
    def cache_batch_size(self) -> Optional[int]:
        return self._cache_batch_size

    @cache_batch_size.setter
    def cache_batch_size(self, val):
        self.set_integration_parameters(cache_batch_size=val)

    @property
    def integration_batch_size(self) -> Optional[int]:
        return self._integration_batch_size

    @integration_batch_size.setter
    def integration_batch_size(self, val):
        self.set_integration_parameters(integration_batch_size=val)

    @property
    def offset(self) -> Optional[float]:
        return self._offset

    @offset.setter
    def offset(self, val):
        self.set_integration_parameters(offset=val)

    @property
    def show_progress(self) -> bool:
        return self._show_progress

    @show_progress.setter
    def show_progress(self, val: bool):
        if not isinstance(val, bool):
            raise ValueError("show_progress must be a boolean")
        self._show_progress = val

    def _check_memory_savings(self):
        inefficient = self._integration_batch_size > (
            self._n_events * self.total_density_integration_nodes / 2
        )
        if inefficient & self._reduce_memory:
            logger.warning(
                f"Since integration_batch_size is too large reduce_memory will increase the memory usage! Disable it if this behavior is not desired."
            )

    @property
    def reduce_memory(self) -> bool:
        return self._reduce_memory

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
                self._irf_energy_node_cache = np.asarray(
                    self._irf_energy_node_cache, dtype=np.float64
                )
        else:
            if self._irf_cache is not None:
                self._irf_cache = torch.as_tensor(self._irf_cache, dtype=torch.float32)
            if self._irf_energy_node_cache is not None:
                self._irf_energy_node_cache = np.asarray(
                    self._irf_energy_node_cache, dtype=np.float32
                )

    @property
    def total_expectation_integration_nodes(self) -> int:
        total = 0
        for x in self._energy_range:
            if isinstance(x, float):
                total += 1
            else:
                total += self._total_expectation_integration_nodes(
                    self._total_expectation_resolution, x[1] - x[0]
                )
        return total

    @property
    def total_density_integration_nodes(self) -> int:
        total = 0
        for x in self._energy_range:
            if isinstance(x, float):
                total += 1
        return sum(self._density_integration_nodes) + total

    def _total_expectation_integration_nodes(
        self, resolution: float, diff: float
    ) -> int:
        return int(np.ceil(diff / resolution)) + 1

    def _set_integration_ranges(
        self,
        density_integration_nodes: Optional[DensityNodeList] = None,
        peak_nodes: Optional[PeakNodeList] = None,
        energy_range: Optional[IEnergyList] = None,
    ):
        if energy_range is not None:
            if not isinstance(energy_range, list):
                raise ValueError("The energy range must be a list.")
            else:
                energy_range_list = []
                for x in energy_range:
                    if not (
                        (isinstance(x, tuple) and len(x) == 2)
                        or isinstance(x, (float, int, np.number))
                    ):
                        raise ValueError(
                            "Each element in the energy range must be either a tuple of length 2 or a float."
                        )
                    else:
                        energy_range_list.append(
                            [float(x[0]), float(x[1])]
                            if isinstance(x, tuple)
                            else float(x)
                        )
                n_intervals = sum(
                    1 for x in energy_range if isinstance(x, tuple) and len(x) == 2
                )
        else:
            n_intervals = self._n_intervals
            energy_range_list = None

        if density_integration_nodes is not None:
            if isinstance(density_integration_nodes, (int, np.integer)):
                density_integration_nodes_list = [
                    int(density_integration_nodes)
                ] * n_intervals
            else:
                if (
                    not isinstance(density_integration_nodes, (list, tuple))
                    or len(density_integration_nodes) != n_intervals
                    or not all(
                        isinstance(n, (int, np.integer))
                        for n in density_integration_nodes
                    )
                ):
                    raise ValueError(
                        "If density_integration_nodes is not an integer, it must be a list of integers of the same length as the number of energy intervals."
                    )
                else:
                    density_integration_nodes_list = [
                        int(n) for n in density_integration_nodes
                    ]
        else:
            if energy_range is not None and hasattr(self, "_density_integration_nodes"):
                if n_intervals == 0:
                    density_integration_nodes_list = []
                elif len(self._density_integration_nodes) == 1:
                    density_integration_nodes_list = [
                        int(self._density_integration_nodes[0])
                    ] * n_intervals
                elif len(self._density_integration_nodes) == n_intervals:
                    density_integration_nodes_list = [
                        int(n) for n in self._density_integration_nodes
                    ]
                else:
                    raise ValueError(
                        f"Energy range changed to {n_intervals} intervals, but current density_integration_nodes has length {len(self._density_integration_nodes)}. Please provide a matching list."
                    )
            else:
                density_integration_nodes_list = None

        if peak_nodes is not None:
            if (
                isinstance(peak_nodes, tuple)
                and len(peak_nodes) == 2
                and all(isinstance(x, (int, np.integer)) for x in peak_nodes)
            ):
                peak_nodes_list = [
                    [int(peak_nodes[0]), int(peak_nodes[1])]
                ] * n_intervals
            else:
                if not all(
                    isinstance(x, tuple)
                    and len(x) == 2
                    and all(isinstance(y, (int, np.integer)) for y in x)
                    for x in peak_nodes
                ) or (len(peak_nodes) != n_intervals):
                    raise ValueError(
                        "Each element in peak_nodes must be a tuple of length 2 containing integers, totalling to the number of energy intervals."
                    )
                peak_nodes_list = [list(int(y) for y in x) for x in peak_nodes]
        else:
            if energy_range is not None and hasattr(self, "_peak_nodes"):
                if n_intervals == 0:
                    peak_nodes_list = []
                elif len(self._peak_nodes) == 1:
                    peak_nodes_list = [list(self._peak_nodes[0])] * n_intervals
                elif len(self._peak_nodes) == n_intervals:
                    peak_nodes_list = [list(x) for x in self._peak_nodes]
                else:
                    raise ValueError(
                        f"Energy range changed to {n_intervals} intervals, but current peak_nodes has length {len(self._peak_nodes)}. Please provide a matching list."
                    )
            else:
                peak_nodes_list = None

        return (
            density_integration_nodes_list,
            peak_nodes_list,
            energy_range_list,
            n_intervals,
        )

    def set_integration_parameters(
        self,  # TODO: Check that intervals dont overlap
        density_integration_nodes: Optional[DensityNodeList] = None,
        total_expectation_resolution: float = -1.0,
        peak_nodes: Optional[PeakNodeList] = None,
        peak_widths: Optional[Tuple[float, float]] = None,
        energy_range: Optional[IEnergyList] = None,
        cache_batch_size: Optional[int] = -1,
        integration_batch_size: Optional[int] = -1,
        offset: float = -1.0,
    ):

        density_integration_nodes, peak_nodes, energy_range, n_intervals = (
            self._set_integration_ranges(
                density_integration_nodes, peak_nodes, energy_range
            )
        )

        new_density_integration_nodes = (
            density_integration_nodes
            if density_integration_nodes is not None
            else self._density_integration_nodes
        )
        new_total_expectation_resolution = (
            total_expectation_resolution
            if total_expectation_resolution != -1.0
            else self._total_expectation_resolution
        )
        new_peak_nodes = peak_nodes if peak_nodes is not None else self._peak_nodes
        new_peak_widths = peak_widths if peak_widths is not None else self._peak_widths
        new_range = energy_range if energy_range is not None else self._energy_range
        new_cache_batch = (
            cache_batch_size if cache_batch_size != -1 else self._cache_batch_size
        )
        new_integration_batch = (
            integration_batch_size
            if integration_batch_size != -1
            else self._integration_batch_size
        )
        new_offset = offset if offset != -1.0 else self._offset

        irf_affected = (
            new_peak_nodes != self._peak_nodes
            or new_peak_widths != self._peak_widths
            or new_density_integration_nodes != self._density_integration_nodes
            or new_range != self._energy_range
        )

        area_affected = (
            new_total_expectation_resolution != self._total_expectation_resolution
            or new_range != self._energy_range
        )

        if irf_affected:
            self._irf_cache = self._irf_energy_node_cache = self._width_tensor = None
            self._nodes_primary = self._nodes_secondary = None
            self._nodes_bkg_0 = self._nodes_bkg_1 = self._nodes_bkg_2 = (
                self._nodes_bkg_3
            ) = None

        if area_affected:
            self._area_cache = self._area_energy_node_cache = None

        if n_intervals > 0:
            if any(
                new_density_integration_nodes[i]
                < (new_peak_nodes[i][0] + 2 * new_peak_nodes[i][1] + 3)
                for i in range(n_intervals)
            ):
                raise ValueError(
                    "Too many nodes per peak compared to the total number or peaks!"
                )

            if any(n < 1 for n in new_density_integration_nodes) or any(
                n < 1 for n in np.hstack(new_peak_nodes)
            ):
                raise ValueError("The number of energy nodes must be at least 1.")

        energy_intervals = [i for i in new_range if isinstance(i, list)]

        if n_intervals > 0:
            smallest_interval = min(
                (energy_intervals[i][1] - energy_intervals[i][0])
                for i in range(n_intervals)
            )
            if (new_total_expectation_resolution > smallest_interval) or (
                new_total_expectation_resolution <= 0
            ):
                raise ValueError(
                    "The total expectation resolution must be positive and smaller than the energy range."
                )
            if any(i[1] <= i[0] for i in energy_intervals):
                raise ValueError("The initial energy interval needs to be increasing!")
        else:
            if new_total_expectation_resolution <= 0:
                raise ValueError("The total expectation resolution must be positive.")

        new_total_expectation_integration_nodes = sum(
            self._total_expectation_integration_nodes(
                new_total_expectation_resolution, i[1] - i[0]
            )
            for i in energy_intervals
        )
        new_max_nodes = max(
            new_total_expectation_integration_nodes, sum(new_density_integration_nodes)
        ) + (len(new_range) - n_intervals)

        if (new_cache_batch is not None) and (new_cache_batch < new_max_nodes):
            raise ValueError(
                f"The cache batch size cannot be smaller than the number of integration nodes ({new_max_nodes})."
            )

        if (new_integration_batch is not None) and (
            new_integration_batch < new_max_nodes
        ):
            raise ValueError(
                f"The integration batch size cannot be smaller than the number of integration nodes ({new_max_nodes})."
            )

        if (new_offset is not None) and (new_offset < 0):
            raise ValueError("The offset cannot be negative.")

        self._density_integration_nodes = new_density_integration_nodes
        self._total_expectation_resolution = new_total_expectation_resolution
        self._peak_nodes = new_peak_nodes
        self._peak_widths = new_peak_widths
        self._energy_range = new_range
        self._cache_batch_size = (
            new_cache_batch
            if new_cache_batch is not None
            else (self._n_events * new_max_nodes)
        )
        self._integration_batch_size = (
            new_integration_batch
            if new_integration_batch is not None
            else (self._n_events * new_max_nodes)
        )
        self._offset = new_offset
        self._n_intervals = n_intervals
        self._check_memory_savings()

    @staticmethod
    def _build_nodes(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, w = np.polynomial.legendre.leggauss(degree)
        return torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), torch.as_tensor(
            w, dtype=torch.float32
        ).unsqueeze(0)

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

        for k in range(len(self._density_integration_nodes)):
            n_density = self._density_integration_nodes[k]
            p_nodes = self._peak_nodes[k]
            w = self._peak_widths

            w_tensor = torch.tensor([w[0], w[0], w[1], w[1]], dtype=torch.float32)
            self._width_tensor.append(w_tensor)

            self._nodes_primary.append(self._build_nodes(p_nodes[0]))
            self._nodes_secondary.append(self._build_nodes(p_nodes[1]))

            self._nodes_bkg_0.append(self._build_nodes(n_density))

            self._nodes_bkg_1.append(self._build_nodes(n_density - p_nodes[0]))

            self._nodes_bkg_2.append(
                (
                    self._build_split_nodes(
                        n_density - p_nodes[0] - p_nodes[1], 2
                    ),  # has_photopeak
                    self._build_split_nodes(
                        n_density - p_nodes[1], 2
                    ),  # ~has_photopeak
                )
            )

            self._nodes_bkg_3.append(
                (
                    self._build_split_nodes(
                        n_density - p_nodes[0] - 2 * p_nodes[1], 3
                    ),  # has_photopeak
                    self._build_split_nodes(
                        n_density - 2 * p_nodes[1], 3
                    ),  # ~has_photopeak
                )
            )

    @staticmethod
    def _scale_nodes_exp(
        E1: torch.Tensor,
        E2: torch.Tensor,
        nodes_u: torch.Tensor,
        weights_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = E2 - E1

        out_n = (nodes_u + 1).mul(0.5).pow(2).mul(diff).add(E1)
        out_w = (nodes_u + 1).mul(0.5).mul(weights_u).mul(diff)

        return out_n, out_w

    @staticmethod
    def _scale_nodes_center(
        E1: torch.Tensor,
        E2: torch.Tensor,
        EC: torch.Tensor,
        nodes_u: torch.Tensor,
        weights_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_left = nodes_u < 0
        width_left = EC - E1
        width_right = E2 - EC

        scale = torch.where(mask_left, width_left, width_right)

        out_n = nodes_u.pow(3).mul(scale).add(EC)
        out_w = nodes_u.pow(2).mul(3).mul(weights_u).mul(scale)

        return out_n, out_w

    @staticmethod
    def _scale_nodes_log(
        E1: torch.Tensor,
        E2: torch.Tensor,
        nodes_u: torch.Tensor,
        weights_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_E1 = torch.log10(E1)
        log_E2 = torch.log10(E2)
        scale = 0.5 * (log_E2 - log_E1)

        out_n = torch.pow(10, nodes_u.mul(scale).add(0.5 * (log_E1 + log_E2)))
        out_w = out_n.mul(weights_u).mul(scale).mul(torch.log(torch.tensor(10.0)))

        return out_n, out_w

    def _get_escape_peak(
        self, energy_m_keV: torch.Tensor, phi_rad: torch.Tensor
    ) -> torch.Tensor:
        E2 = 511.0 / (1.0 + 511.0 / energy_m_keV - torch.cos(phi_rad))
        energy = energy_m_keV + 1022.0 - E2

        accept = (energy > 1600.0) & (energy_m_keV < energy)
        return torch.where(
            accept, energy, torch.tensor(float("nan"), dtype=torch.float32)
        )

    def _get_missing_energy_peak(
        self,
        phi_geo_rad: torch.Tensor,
        energy_m_keV: torch.Tensor,
        phi_rad: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        cos_geo = torch.cos(phi_geo_rad)
        cos_phi = torch.cos(phi_rad)

        if inverse:
            denom = (
                2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            )
            root = torch.sqrt(
                energy_m_keV
                * (cos_geo - 1)
                * (
                    -2044.0
                    - 5 * energy_m_keV
                    + energy_m_keV * cos_geo
                    + 4 * energy_m_keV * cos_phi
                )
            )
            energy = 511.0 * (energy_m_keV - energy_m_keV * cos_geo + root) / denom
        else:
            denom = (
                2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            )
            root = torch.sqrt(
                energy_m_keV**2
                * (cos_geo - 1)
                * (cos_phi - 1)
                * (
                    (1022.0 + energy_m_keV) ** 2
                    - energy_m_keV * (2044.0 + energy_m_keV) * cos_phi
                    - 2 * energy_m_keV**2 * cos_geo * torch.sin(phi_rad / 2) ** 2
                )
            )
            energy = (
                energy_m_keV**2 * (1 - cos_geo - cos_phi + cos_phi * cos_geo) + root
            ) / denom

        accept = (energy > energy_m_keV) & (energy_m_keV / energy - 1 < -0.2)
        return torch.where(
            accept, energy, torch.tensor(float("nan"), dtype=torch.float32)
        )

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

        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None

    def set_source(self, source: Source):
        if not isinstance(source, ExtendedSource):
            raise TypeError("Please provide a ExtendedSource!")

        self._source = source

    def copy(self) -> CachedUnbinnedThreeMLSourceResponseInterface:
        new_instance = copy.copy(self)
        new_instance.clear_cache()
        new_instance._source = None

        return new_instance

    @staticmethod
    def _earth_occ(source_coord: SkyCoord, ori: SpacecraftHistory) -> np.ndarray:
        dist_earth_center = ori.location.spherical.distance.km
        max_angle = np.pi - np.arcsin(c.R_earth.to(u.km).value / dist_earth_center)
        src_angle = source_coord.separation(ori.earth_zenith)
        return (src_angle.to(u.rad).value < max_angle).astype(np.float32)

    @staticmethod
    def _get_target_in_sc_frame(
        source_coord: SkyCoord, ori: SpacecraftHistory
    ) -> SkyCoord:
        src_in_sc_frame = SkyCoord(
            np.dot(
                ori.attitude.rot.inv().as_matrix(),
                source_coord.transform_to(ori.attitude.frame).cartesian.xyz.value,
            ),
            representation_type="cartesian",
            frame=SpacecraftFrame(),
        )

        src_in_sc_frame.representation_type = "spherical"
        return src_in_sc_frame

    def _compute_area(self):
        n_energy = self.total_expectation_integration_nodes

        e_n, e_w = [], []

        for x in self._energy_range:
            if isinstance(x, (float, int)):
                e_n.append([float(x)])
                e_w.append([1.0])
            else:
                E1, E2 = torch.tensor(x[0]), torch.tensor(x[1])
                n_nodes = self._total_expectation_integration_nodes(
                    self._total_expectation_resolution, E2 - E1
                )

                n_np, w_np = np.polynomial.legendre.leggauss(n_nodes)

                n_torch = torch.from_numpy(n_np)
                w_torch = torch.from_numpy(w_np)

                n, w = self._scale_nodes_log(E1, E2, n_torch, w_torch)

                e_n.append(n.numpy())
                e_w.append(w.numpy())

        self._area_energy_node_cache = np.concatenate(e_n).astype(np.float64)
        e_n = self._area_energy_node_cache.astype(np.float32)
        e_w = np.concatenate(e_w).astype(np.float32)

        #the eff area have dimension (nb pixel , nb energy nodes)
        total_area = np.zeros( (len(self._pixels),n_energy), dtype=np.float64)
        
        #Need to loop on every pixels
        for pix_ind, pix in enumerate(tqdm(self._pixels, 
                      disable=(not self.show_progress),
                      desc="Caching the total effective area", 
                      smoothing=0.2, 
                      leave=False)):
        
            
        
            # Midpoint
            sc_coord_sph = self._get_target_in_sc_frame(self._source_coord_cache[pix_ind], self._sc_ori_center)
            earth_occ_index = self._earth_occ(self._source_coord_cache[pix_ind], self._sc_ori_center)

            combined_time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(
                np.float32
            ) * earth_occ_index

            # Simpson
            # sc_coord_sph = self._sc_ori_simpson.get_target_in_sc_frame(coord)
            # earth_occ_index = self._earth_occ(coord, self._sc_ori_simpson)

            # combined_time_weights = (self._unique_time_weights * earth_occ_index).astype(np.float32)

            lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
            lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

            n_time = len(lon_ph_rad)
            batch_size_time = self._cache_batch_size // n_energy
    
    
            max_batch_total = n_energy * min(batch_size_time, n_time)
            batch_lons_buffer = np.empty(max_batch_total, dtype=np.float32)
            batch_lats_buffer = np.empty(max_batch_total, dtype=np.float32)
            batch_energies_buffer = np.empty(max_batch_total, dtype=np.float32)
    
            for i in range(0, n_time, batch_size_time):
                start = i
                end = min(i + batch_size_time, n_time)
                current_n_time = end - start
                current_total = current_n_time * n_energy
    
                batch_lons_buffer[:current_total].reshape(current_n_time, n_energy)[:] = (
                    lon_ph_rad[start:end, np.newaxis]
                )
                batch_lats_buffer[:current_total].reshape(current_n_time, n_energy)[:] = (
                    lat_ph_rad[start:end, np.newaxis]
                )
                batch_energies_buffer[:current_total].reshape(current_n_time, n_energy)[
                    :
                ] = e_n
    
                photons = PhotonListWithDirectionAndEnergyInSCFrame(
                    batch_lons_buffer[:current_total],
                    batch_lats_buffer[:current_total],
                    batch_energies_buffer[:current_total],
                )
    
                eff_areas_flat = asarray(
                    self._irf.effective_area_cm2(photons), dtype=np.float32
                )
                eff_areas_grid = eff_areas_flat.reshape(current_n_time, n_energy)
    
                total_area[pix_ind] += np.einsum(
                    "ij,i,j->j",
                    eff_areas_grid,
                    combined_time_weights[start:end],
                    e_w.ravel(),
                ) * self._pixelsolidang

        self._area_cache = total_area

    def _fill_nodes(
        self,
        nodes_out: torch.Tensor,
        weights_out: torch.Tensor,
        indices: torch.Tensor,
        mode: int,
        has_photopeak: torch.Tensor,
        sorted_peaks: torch.Tensor,
        delta: torch.Tensor,
        intervals_idx: int,
        current_offset: int,
        Emin: float,
        Emax: float,
    ):

        Emin_t = torch.full((len(indices), 1), Emin, dtype=torch.float32)
        Emax_t = torch.full((len(indices), 1), Emax, dtype=torch.float32)

        if mode == 0:
            # [Background]
            c = 0
            w = self._nodes_bkg_0[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(
                Emin_t, Emax_t, *self._nodes_bkg_0[intervals_idx]
            )
            nodes_out[indices, current_offset + c : current_offset + c + w] = n_res
            weights_out[indices, current_offset + c : current_offset + c + w] = w_res

        elif mode == 1:
            E1 = (sorted_peaks[:, 0] - delta[:, 0]).clamp(min=Emin)
            E2 = (sorted_peaks[:, 0] + delta[:, 0]).clamp(max=Emax)

            EC = sorted_peaks[:, 0]

            E1, E2, EC = [E.view(-1, 1) for E in (E1, E2, EC)]

            # [Photopeak + Background]
            if torch.any(has_photopeak):
                c = 0
                w = self._nodes_primary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(
                    E1[has_photopeak],
                    E2[has_photopeak],
                    EC[has_photopeak],
                    *self._nodes_primary[intervals_idx],
                )
                nodes_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_bkg_1[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    E2[has_photopeak],
                    Emax_t[has_photopeak],
                    *self._nodes_bkg_1[intervals_idx],
                )
                nodes_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

            # [Background + Secondary Peak + Background]
            if torch.any(~has_photopeak):
                c = 0
                w = self._nodes_bkg_2[intervals_idx][1][0][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    Emin_t[~has_photopeak],
                    E1[~has_photopeak],
                    *self._nodes_bkg_2[intervals_idx][1][0],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(
                    E1[~has_photopeak],
                    E2[~has_photopeak],
                    EC[~has_photopeak],
                    *self._nodes_secondary[intervals_idx],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_bkg_2[intervals_idx][1][1][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    E2[~has_photopeak],
                    Emax_t[~has_photopeak],
                    *self._nodes_bkg_2[intervals_idx][1][1],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

        elif mode == 2:
            center_peak = (sorted_peaks[:, 0] + sorted_peaks[:, 1]) / 2

            E1 = (sorted_peaks[:, 0] - delta[:, 0]).clamp(min=Emin)
            E3 = (sorted_peaks[:, 1] - delta[:, 1]).clamp(min=center_peak)
            E2 = (sorted_peaks[:, 0] + delta[:, 0]).clamp(max=E3)
            E4 = (sorted_peaks[:, 1] + delta[:, 1]).clamp(max=Emax)

            EC1 = sorted_peaks[:, 0]
            EC2 = sorted_peaks[:, 1]

            E1, E2, E3, E4, EC1, EC2 = [
                E.view(-1, 1) for E in (E1, E2, E3, E4, EC1, EC2)
            ]

            # [Photopeak + Background + Secondary Peak + Background]
            if torch.any(has_photopeak):
                c = 0
                w = self._nodes_primary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(
                    E1[has_photopeak],
                    E2[has_photopeak],
                    EC1[has_photopeak],
                    *self._nodes_primary[intervals_idx],
                )
                nodes_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_bkg_2[intervals_idx][0][0][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    E2[has_photopeak],
                    E3[has_photopeak],
                    *self._nodes_bkg_2[intervals_idx][0][0],
                )
                nodes_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(
                    E3[has_photopeak],
                    E4[has_photopeak],
                    EC2[has_photopeak],
                    *self._nodes_secondary[intervals_idx],
                )
                nodes_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_bkg_2[intervals_idx][0][1][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    E4[has_photopeak],
                    Emax_t[has_photopeak],
                    *self._nodes_bkg_2[intervals_idx][0][1],
                )
                nodes_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

            # [Background + Secondary Peak + Background + Secondary Peak + Background]
            if torch.any(~has_photopeak):
                c = 0
                w = self._nodes_bkg_3[intervals_idx][1][0][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    Emin_t[~has_photopeak],
                    E1[~has_photopeak],
                    *self._nodes_bkg_3[intervals_idx][1][0],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(
                    E1[~has_photopeak],
                    E2[~has_photopeak],
                    EC1[~has_photopeak],
                    *self._nodes_secondary[intervals_idx],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_bkg_3[intervals_idx][1][1][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    E2[~has_photopeak],
                    E3[~has_photopeak],
                    *self._nodes_bkg_3[intervals_idx][1][1],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_secondary[intervals_idx][0].shape[1]
                n_res, w_res = self._scale_nodes_center(
                    E3[~has_photopeak],
                    E4[~has_photopeak],
                    EC2[~has_photopeak],
                    *self._nodes_secondary[intervals_idx],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

                c += w
                w = self._nodes_bkg_3[intervals_idx][1][2][0].shape[1]
                n_res, w_res = self._scale_nodes_exp(
                    E4[~has_photopeak],
                    Emax_t[~has_photopeak],
                    *self._nodes_bkg_3[intervals_idx][1][2],
                )
                nodes_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = n_res
                weights_out[
                    indices[~has_photopeak], current_offset + c : current_offset + c + w
                ] = w_res

        elif mode == 3:
            center_peak_1 = (sorted_peaks[:, 0] + sorted_peaks[:, 1]) / 2
            center_peak_2 = (sorted_peaks[:, 1] + sorted_peaks[:, 2]) / 2

            E1 = (sorted_peaks[:, 0] - delta[:, 0]).clamp(min=Emin)
            E3 = (sorted_peaks[:, 1] - delta[:, 1]).clamp(min=center_peak_1)
            E2 = (sorted_peaks[:, 0] + delta[:, 0]).clamp(max=E3)
            E4 = (sorted_peaks[:, 1] + delta[:, 1]).clamp(max=center_peak_2)
            E5 = (sorted_peaks[:, 2] - delta[:, 2]).clamp(min=E4)
            E6 = (sorted_peaks[:, 2] + delta[:, 2]).clamp(max=Emax)

            EC1, EC2, EC3 = [sorted_peaks[:, i] for i in range(3)]

            E1, E2, E3, E4, E5, E6, EC1, EC2, EC3 = [
                E.view(-1, 1) for E in (E1, E2, E3, E4, E5, E6, EC1, EC2, EC3)
            ]

            # [Photopeak + Background + Secondary Peak + Background + Secondary Peak + Background]
            c = 0
            w = self._nodes_primary[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_center(
                E1[has_photopeak],
                E2[has_photopeak],
                EC1[has_photopeak],
                *self._nodes_primary[intervals_idx],
            )
            nodes_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = n_res
            weights_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = w_res

            c += w
            w = self._nodes_bkg_3[intervals_idx][0][0][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(
                E2[has_photopeak],
                E3[has_photopeak],
                *self._nodes_bkg_3[intervals_idx][0][0],
            )
            nodes_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = n_res
            weights_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = w_res

            c += w
            w = self._nodes_secondary[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_center(
                E3[has_photopeak],
                E4[has_photopeak],
                EC2[has_photopeak],
                *self._nodes_secondary[intervals_idx],
            )
            nodes_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = n_res
            weights_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = w_res

            c += w
            w = self._nodes_bkg_3[intervals_idx][0][1][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(
                E4[has_photopeak],
                E5[has_photopeak],
                *self._nodes_bkg_3[intervals_idx][0][1],
            )
            nodes_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = n_res
            weights_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = w_res

            c += w
            w = self._nodes_secondary[intervals_idx][0].shape[1]
            n_res, w_res = self._scale_nodes_center(
                E5[has_photopeak],
                E6[has_photopeak],
                EC3[has_photopeak],
                *self._nodes_secondary[intervals_idx],
            )
            nodes_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = n_res
            weights_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = w_res

            c += w
            w = self._nodes_bkg_3[intervals_idx][0][2][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(
                E6[has_photopeak],
                Emax_t[has_photopeak],
                *self._nodes_bkg_3[intervals_idx][0][2],
            )
            nodes_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = n_res
            weights_out[
                indices[has_photopeak], current_offset + c : current_offset + c + w
            ] = w_res
        else:
            raise ValueError(f"Unknown folding mode {mode}")

    def _get_nodes(
        self,
        energy_m_keV: torch.Tensor,
        phi_rad: torch.Tensor,
        phi_geo_rad: torch.Tensor,
        phi_igeo_rad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        energy_m_keV = energy_m_keV.view(-1, 1)
        phi_rad = phi_rad.view(-1, 1)
        phi_geo_rad = phi_geo_rad.view(-1, 1)
        phi_igeo_rad = phi_igeo_rad.view(-1, 1)

        batch_size = energy_m_keV.shape[0]
        total_nodes = self.total_density_integration_nodes

        nodes = torch.zeros((batch_size, total_nodes), dtype=torch.float32)
        weights = torch.zeros_like(nodes)

        raw_peaks = torch.zeros((batch_size, 4), dtype=torch.float32)
        raw_peaks[:, 0] = energy_m_keV.squeeze()
        raw_peaks[:, 1] = self._get_escape_peak(energy_m_keV, phi_rad).squeeze()
        raw_peaks[:, 2] = self._get_missing_energy_peak(
            phi_geo_rad, energy_m_keV, phi_rad
        ).squeeze()
        raw_peaks[:, 3] = self._get_missing_energy_peak(
            phi_igeo_rad, energy_m_keV, phi_rad, inverse=True
        ).squeeze()

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

            interval_peaks = torch.where(
                (raw_peaks >= Emin) & (raw_peaks <= Emax),
                raw_peaks,
                torch.tensor(float("nan"), dtype=torch.float32),
            )

            w_tensor = self._width_tensor[intervals_idx]
            interval_diffs = interval_peaks * w_tensor[None, ...]

            n_peaks = torch.sum(~torch.isnan(interval_peaks), dim=1)

            for mode in range(4):
                indices = torch.where(n_peaks == mode)[0]
                if len(indices) == 0:
                    continue

                has_photopeak = ~torch.isnan(interval_peaks[:, 0][indices])
                p_sub = interval_peaks[indices]
                d_sub = interval_diffs[indices]

                if mode == 0:
                    self._fill_nodes(
                        nodes,
                        weights,
                        indices,
                        0,
                        has_photopeak,
                        None,
                        None,
                        intervals_idx,
                        current_offset,
                        Emin,
                        Emax,
                    )
                elif mode == 1:
                    mask = ~torch.isnan(p_sub)
                    p_comp = p_sub[mask].view(-1, 1)
                    d_comp = d_sub[mask].view(-1, 1)

                    self._fill_nodes(
                        nodes,
                        weights,
                        indices,
                        1,
                        has_photopeak,
                        p_comp,
                        d_comp,
                        intervals_idx,
                        current_offset,
                        Emin,
                        Emax,
                    )
                elif mode == 2:
                    mask = ~torch.isnan(p_sub)
                    p_comp = p_sub[mask].view(-1, 2)
                    d_comp = d_sub[mask].view(-1, 2)

                    p_sorted, idx = torch.sort(p_comp, dim=1)
                    d_sorted = torch.gather(d_comp, 1, idx)

                    self._fill_nodes(
                        nodes,
                        weights,
                        indices,
                        2,
                        has_photopeak,
                        p_sorted,
                        d_sorted,
                        intervals_idx,
                        current_offset,
                        Emin,
                        Emax,
                    )
                elif mode == 3:
                    mask = ~torch.isnan(p_sub)
                    p_comp = p_sub[mask].view(-1, 3)
                    d_comp = d_sub[mask].view(-1, 3)

                    p_sorted, idx = torch.sort(p_comp, dim=1)
                    d_sorted = torch.gather(d_comp, 1, idx)

                    self._fill_nodes(
                        nodes,
                        weights,
                        indices,
                        3,
                        has_photopeak,
                        p_sorted,
                        d_sorted,
                        intervals_idx,
                        current_offset,
                        Emin,
                        Emax,
                    )

            current_offset += size
            intervals_idx += 1

        return nodes, weights

        # diffs = peaks * self._width_tensor[None, ...]
        #
        # n_peaks = torch.sum(~torch.isnan(peaks), dim=1)
        #
        # indices_1 = torch.where(n_peaks == 1)[0]
        # indices_2 = torch.where(n_peaks == 2)[0]
        # indices_3 = torch.where(n_peaks == 3)[0]
        #
        # if len(indices_1) > 0:
        #    self._fill_nodes(nodes, weights, indices_1, 1,
        #                              peaks[indices_1, :1], diffs[indices_1, :1])
        #
        # if len(indices_2) > 0:
        #    p_sub = peaks[indices_2]
        #    d_sub = diffs[indices_2]
        #    mask = ~torch.isnan(p_sub)
        #    p_comp = p_sub[mask].view(-1, 2)
        #    d_comp = d_sub[mask].view(-1, 2)
        #    self._fill_nodes(nodes, weights, indices_2, 2, p_comp, d_comp)
        #
        # if len(indices_3) > 0:
        #    p_sub = peaks[indices_3]
        #    d_sub = diffs[indices_3]
        #    mask = ~torch.isnan(p_sub)
        #    p_comp = p_sub[mask].view(-1, 3)
        #    d_comp = d_sub[mask].view(-1, 3)
        #
        #    p_sorted, idx = torch.sort(p_comp, dim=1)
        #    d_sorted = torch.gather(d_comp, 1, idx)
        #
        #    self._fill_nodes(nodes, weights, indices_3, 3, p_sorted, d_sorted)
        #
        # return nodes, weights

    def _get_CDS_coordinates(
        self, lon_src_rad: torch.Tensor, lat_src_rad: torch.Tensor, indices=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_lat_src = torch.cos(lat_src_rad)
        sin_lat_src = torch.sin(lat_src_rad)
        cos_lon_src = torch.cos(lon_src_rad)
        sin_lon_src = torch.sin(lon_src_rad)

        if indices is None:
            indices = slice(0, len(self._cos_lat_scatt))

        cos_geo = (
            cos_lat_src
            * cos_lon_src
            * self._cos_lat_scatt[indices]
            * self._cos_lon_scatt[indices]
            + cos_lat_src
            * sin_lon_src
            * self._cos_lat_scatt[indices]
            * self._sin_lon_scatt[indices]
            + sin_lat_src * self._sin_lat_scatt[indices]
        )

        cos_geo = torch.clip(cos_geo, -1.0, 1.0)
        phi_geo_rad = torch.arccos(cos_geo)

        return phi_geo_rad, np.pi - phi_geo_rad

    def _compute_nodes(self, pix):
        
        sc_coord_sph = self._sc_coord_sph_cache[pix][self._valid_mask_cache[pix]]

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        
        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(
            torch.as_tensor(lon_ph_rad),
            torch.as_tensor(lat_ph_rad),
            indices=self._valid_mask_cache[pix],
        )
        np_memory_dtype = np.float32 if self._reduce_memory else np.float64
        self._irf_energy_node_cache.append( np.asarray(
            self._get_nodes(
                self._energy_m_keV[self._valid_mask_cache[pix]],
                self._phi_rad[self._valid_mask_cache[pix]],
                phi_geo_rad,
                phi_igeo_rad,
            )[0],
            dtype=np_memory_dtype,
        ))

    def _compute_density_helper(
        self,
        indices: np.ndarray,
        source_coord,
        sc_coord_sph=None,
        earth_occ_index: Optional[np.ndarray] = None,
        buffer: Optional[
            Tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
            ]
        ] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        indices = np.asarray(indices)

        if sc_coord_sph is None:
            sc_coord_sph = self._get_target_in_sc_frame(
                source_coord, self._sc_ori_unique
            )[self._inv_idx[indices]]
        else:
            sc_coord_sph = sc_coord_sph[indices]
        if earth_occ_index is None:
            earth_occ_index = self._earth_occ(source_coord, self._sc_ori_unique)[
                self._inv_idx[indices]
            ]
        else:
            earth_occ_index = earth_occ_index[indices]

        live_sub = self._livetime_ratio[indices]
        current_n = len(earth_occ_index)

        e_sl = self._energy_m_keV[indices]
        p_sl = self._phi_rad[indices]

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(
            torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad), indices=indices
        )

        nodes, weights = self._get_nodes(e_sl, p_sl, phi_geo_rad, phi_igeo_rad)
        n_energy = self.total_density_integration_nodes

        current_total = current_n * n_energy

        if buffer is not None:
            (
                batch_lon_src_buffer,
                batch_lat_src_buffer,
                batch_energy_buffer,
                batch_phi_buffer,
                batch_lon_scatt_buffer,
                batch_lat_scatt_buffer,
            ) = buffer
        else:
            buffer_size = n_energy * current_n
            batch_lon_src_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lat_src_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_energy_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_phi_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lon_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lat_scatt_buffer = np.empty(buffer_size, dtype=np.float32)

        batch_lon_src_buffer[:current_total].reshape(current_n, n_energy)[:] = (
            lon_ph_rad[:, np.newaxis]
        )
        batch_lat_src_buffer[:current_total].reshape(current_n, n_energy)[:] = (
            lat_ph_rad[:, np.newaxis]
        )

        batch_energy_buffer[:current_total].reshape(current_n, n_energy)[:] = (
            np.asarray(e_sl[:, np.newaxis])
        )
        batch_lon_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = (
            np.asarray(self._lon_scatt[indices, np.newaxis])
        )
        batch_lat_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = (
            np.asarray(self._lat_scatt[indices, np.newaxis])
        )
        batch_phi_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(
            p_sl[:, np.newaxis]
        )

        photons = PhotonListWithDirectionAndEnergyInSCFrame(
            batch_lon_src_buffer[:current_total],
            batch_lat_src_buffer[:current_total],
            np.asarray(nodes).ravel(),
        )

        events = EmCDSEventDataInSCFrameFromArrays(
            batch_energy_buffer[:current_total],
            batch_lon_scatt_buffer[:current_total],
            batch_lat_scatt_buffer[:current_total],
            batch_phi_buffer[:current_total],
        )

        res_block = torch.as_tensor(
            asarray(
                self._irf.differential_effective_area_cm2(photons, events),
                dtype=np.float32,
            )
        ).view(current_n, n_energy)

        occ = torch.as_tensor(earth_occ_index).view(-1, 1)
        live = torch.as_tensor(live_sub).view(-1, 1)

        res_block *= occ * live * weights * self._pixelsolidang

        np_memory_dtype = np.float32 if self._reduce_memory else np.float64
        return res_block, np.asarray(nodes, dtype=np_memory_dtype)

    def _compute_density(self):
        n_energy = self.total_density_integration_nodes
        batch_size_events = self._cache_batch_size // n_energy

        torch_memory_dtype = torch.float32 if self._reduce_memory else torch.float64
        np_memory_dtype = np.float32 if self._reduce_memory else np.float64
        
        #loop on the pixels
        #if mask is None, j == pix
        for j,pix in enumerate(tqdm(self._pixels, 
                      disable=(not self.show_progress),
                      desc="Caching the response", 
                      smoothing=0.2, 
                      leave=False)):
        
            
            earth_occ_index = self._earth_occ(self._source_coord_cache[j], self._sc_ori_center)

            #get the precomputed sc_coord_sph from that pixel
            sc_coord_sph = self._sc_coord_sph_cache[j]
            
            self._valid_events = int(len(self._valid_mask_cache[j]))
    
            self._irf_cache.append( torch.zeros(
                (self._valid_events, n_energy), dtype=torch_memory_dtype
            ))
    
            batched_node_caching = (batch_size_events < self._valid_events) & (
                self._force_energy_node_caching
            )
            if batched_node_caching:
                self._irf_energy_node_cache.append( np.zeros(
                    (self._valid_events, n_energy), dtype=np_memory_dtype
                ))
    
            #
            # lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
            # lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
            #
            # phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad))
            #
            buffer_size = n_energy * min(batch_size_events, self._valid_events)
            batch_lon_src_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lat_src_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_energy_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_phi_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lon_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
            batch_lat_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
    
            buffer = (
                batch_lon_src_buffer,
                batch_lat_src_buffer,
                batch_energy_buffer,
                batch_phi_buffer,
                batch_lon_scatt_buffer,
                batch_lat_scatt_buffer,
            )
    
            for i in range(0, self._valid_events, batch_size_events):
                start = i
                end = min(i + batch_size_events, self._valid_events)
    
                slice_sub = slice(start, end)
    
                res_block, nodes = self._compute_density_helper(
                    self._valid_mask_cache[j][slice_sub],
                    self._source_coord_cache[j],
                    sc_coord_sph,
                    earth_occ_index,
                    buffer,
                )
    
                # current_n = end - start
                # current_total = current_n * n_energy
                #
                # e_sl = self._energy_m_keV[start:end]
                # p_sl = self._phi_rad[start:end]
                # pg_sl = phi_geo_rad[start:end]
                # pig_sl = phi_igeo_rad[start:end]
                #
                # nodes, weights = self._get_nodes(e_sl, p_sl, pg_sl, pig_sl)
    
                if batch_size_events >= self._valid_events:
                    try :
                        self._irf_energy_node_cache[j] = nodes.astype(np_memory_dtype)
                    except: 
                        self._irf_energy_node_cache.append(nodes.astype(np_memory_dtype))

                if batched_node_caching:
                        self._irf_energy_node_cache[j][start:end] = nodes.astype(np_memory_dtype)
    
                self._irf_cache[j][start:end] = res_block.to(torch_memory_dtype)
    
                # batch_lon_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lon_ph_rad[start:end, np.newaxis]
                # batch_lat_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lat_ph_rad[start:end, np.newaxis]
                #
                # batch_energy_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._energy_m_keV[start:end, np.newaxis])
                # batch_lon_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lon_scatt[start:end, np.newaxis])
                # batch_lat_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lat_scatt[start:end, np.newaxis])
                # batch_phi_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._phi_rad[start:end, np.newaxis])
                #
                # photons = PhotonListWithDirectionAndEnergyInSCFrame(
                #    batch_lon_src_buffer[:current_total],
                #    batch_lat_src_buffer[:current_total],
                #    np.asarray(nodes).ravel()
                #    )
                #
                # events = EmCDSEventDataInSCFrameFromArrays(
                #    batch_energy_buffer[:current_total],
                #    batch_lon_scatt_buffer[:current_total],
                #    batch_lat_scatt_buffer[:current_total],
                #    batch_phi_buffer[:current_total],
                # )
    
                # res_block = torch.as_tensor(asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=np.float32)).view(current_n, n_energy)
    
                ###eff_areas_flat = torch.as_tensor(asarray(self._irf._effective_area_cm2(photons), dtype=np.float32))
                ###densities_flat = torch.as_tensor(asarray(self._irf._event_probability(photons, events), dtype=np.float32))
    
                ###res_block = (densities_flat * eff_areas_flat).view(current_n, n_energy)
    
                # occ = torch.as_tensor(earth_occ_index[start:end]).view(-1, 1)
                # live = torch.as_tensor(self._livetime_ratio[start:end]).view(-1, 1)
    
                # res_block *= occ * live * weights
    
                # self._irf_cache[start:end] = res_block

    def _update_cache(self):

        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        #Mask the pixels if given 
        if self._mask is not None:
            self._pixels = self._set_mask()
        else:
            self._pixels = np.arange(self._npix)
            
        lon, lat = hp.pix2ang(self._nside, self._pixels, lonlat=True)
        
        #array of SkyCoord for each pixel
        source_coord = SkyCoord(lon*u.deg, lat*u.deg, frame ="galactic")
        self._lon = lon
        self._lat = lat
        self._source_coord_cache = source_coord
        
        #In order to compare it with source_coord we need to give it 
        #an array dimension
        if self._last_convolved_source_skycoord is None :
            self._last_convolved_source_skycoord = np.zeros(len(source_coord))
            
        if (self._sc_coord_sph_cache is None) or (
            (source_coord != self._last_convolved_source_skycoord).any()
        ):
            self._sc_coord_sph_cache = []
            for pix in range(len(self._pixels)):
                
                self._sc_coord_sph_cache.append( self._get_target_in_sc_frame(
                    source_coord[pix], self._sc_ori_unique
                )[self._inv_idx] )

        
        if (self._valid_mask_cache is None) or (
            (source_coord != self._last_convolved_source_skycoord).any()
        ):
            self._valid_mask_cache = []
            for pix in range(len(self._pixels)):
                earth_occ_index = self._earth_occ(source_coord[pix], self._sc_ori_unique)[self._inv_idx]
                self._valid_mask_cache.append(  np.where(
                (earth_occ_index > 0) & (self._livetime_ratio > 0)
                )[0] )
        
        no_recalculation = (
            (source_coord == self._last_convolved_source_skycoord).all()
            and (self._irf_cache is not None)
            and (self._area_cache is not None)
        )

        area_recalculation = (source_coord != self._last_convolved_source_skycoord).any() or (
            self._area_cache is None
        )

        pdf_recalculation = (source_coord != self._last_convolved_source_skycoord).any() or (
            self._irf_cache is None
        )

        if no_recalculation:
            pass
        else:
            active_pool = True
            if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
                active_pool = self._irf.active_pool
                if not active_pool:
                    self._irf.init_compute_pool()

            if (source_coord != self._last_convolved_source_skycoord).any():
                self._irf_energy_node_cache = None

            if area_recalculation:
                self._area_cache = []
                self._compute_area()

            if pdf_recalculation:
                self._irf_cache = []
                self._irf_energy_node_cache = []
                self._init_node_pool()
                self._compute_density()

            if not active_pool:
                self._irf.shutdown_compute_pool()

            self._last_convolved_source_skycoord = source_coord.copy()

        node_caching = (
            self.force_energy_node_caching and self._irf_energy_node_cache is None
        )

        if node_caching:
            for pix in range(len(self._pixels)):
                self._compute_nodes(pix)

    def cache_to_file(self, filename: Union[str, Path]):
        with h5py.File(str(filename), "w") as f:

            def processed_energy_range():
                e_range = []
                for x in self._energy_range:
                    if isinstance(x, list):
                        e_range.append(x)
                    else:
                        e_range.append([x, x])
                return np.array(e_range, dtype=np.float64)

            f.attrs["total_expectation_resolution"] = self._total_expectation_resolution
            f.attrs["peak_widths"] = self._peak_widths
            f.attrs["cache_batch_size"] = self._cache_batch_size
            f.attrs["integration_batch_size"] = self._integration_batch_size
            f.attrs["show_progress"] = self._show_progress
            f.attrs["force_energy_node_caching"] = self._force_energy_node_caching
            f.attrs["reduce_memory"] = self._reduce_memory
            f.attrs["n_intervals"] = self._n_intervals

            f.create_dataset("energy_range", data=processed_energy_range())
            f.create_dataset(
                "peak_nodes", data=np.array(self._peak_nodes, dtype=np.int32)
            )
            f.create_dataset(
                "density_integration_nodes",
                data=np.array(self._density_integration_nodes, dtype=np.int32),
            )

            if self._offset is not None:
                f.attrs["offset"] = self._offset

            if self._valid_events is not None:
                f.attrs["valid_events"] = self._valid_events

            if self._irf_cache is not None:
                # Glue the list of tensor end-to-end into a single 2D block
                flat_data = torch.cat(self._irf_cache, dim=0).numpy()
                f.create_dataset("irf_cache", data=flat_data, compression="gzip")
                
                # CRITICAL: Save the row counts per pixel so you can reconstruct the list later
                cache_lengths = np.array([t.shape[0] for t in self._irf_cache], dtype=np.int32)
                f.create_dataset("irf_cache_lengths", data=cache_lengths, compression="gzip")

            if self._irf_energy_node_cache is not None:
                # 1. Flatten all elements end-to-end into a single continuous array
                flat_nodes = np.concatenate(self._irf_energy_node_cache)
                f.create_dataset(
                    "irf_energy_node_cache",
                    data=flat_nodes,
                    compression="gzip",
                )
                # 2. Save the individual lengths so you know where to split them on reload
                node_lengths = np.array([len(item) for item in self._irf_energy_node_cache], dtype=np.int32)
                f.create_dataset("irf_energy_node_cache_lengths", data=node_lengths, compression="gzip")
                
            if self._area_cache is not None:
                f.create_dataset(
                    "area_cache", data=self._area_cache, compression="gzip"
                )

            if self._area_energy_node_cache is not None:
                f.create_dataset(
                    "area_energy_node_cache",
                    data=self._area_energy_node_cache,
                    compression="gzip",
                )

            if self._exp_events is not None:
                f.create_dataset("exp_events", data=self._exp_events)

            if self._exp_density is not None:
                f.create_dataset(
                    "exp_density", data=self._exp_density.numpy(), compression="gzip"
                )

            if self._valid_mask_cache is not None:
                
                # 1. Flatten all elements end-to-end into a single continuous array
                flat_nodes = np.concatenate(self._valid_mask_cache)
                f.create_dataset(
                    "valid_mask_cache",
                    data=flat_nodes,
                    compression="gzip",
                )
                # 2. Save the individual lengths so you know where to split them on reload
                node_lengths = np.array([len(item) for item in self._valid_mask_cache], dtype=np.int32)
                f.create_dataset("valid_mask_cache_lengths", data=node_lengths, compression="gzip")

            if self._last_convolved_source_dict_number is not None:
                json_str = json.dumps(self._last_convolved_source_dict_number)
                f.attrs["last_convolved_source_dict_number"] = json_str

            if self._last_convolved_source_dict_density is not None:
                json_str = json.dumps(self._last_convolved_source_dict_density)
                f.attrs["last_convolved_source_dict_density"] = json_str

            if self._last_convolved_source_skycoord is not None:
                sc = self._last_convolved_source_skycoord
                f.attrs["last_convolved_lon_deg"] = sc.spherical.lon.deg
                f.attrs["last_convolved_lat_deg"] = sc.spherical.lat.deg
                f.attrs["last_convolved_frame"] = sc.frame.name
                if hasattr(sc, "equinox") and sc.equinox is not None:
                    f.attrs["last_convolved_equinox"] = sc.equinox.value

    def cache_from_file(self, filename: Union[str, Path]):
        if not os.path.exists(str(filename)):
            raise FileNotFoundError(f"Cache file {str(filename)} not found.")

        with h5py.File(str(filename), "r") as f:

            def processed_energy_range(input):
                e_range = []
                for x in input:
                    if x[0] != x[1]:
                        e_range.append([float(x[0]), float(x[1])])
                    else:
                        e_range.append(float(x[0]))
                return e_range

            self._total_expectation_resolution = float(
                f.attrs["total_expectation_resolution"]
            )
            self._peak_widths = tuple(f.attrs["peak_widths"])
            self._cache_batch_size = int(f.attrs["cache_batch_size"])
            self._integration_batch_size = int(f.attrs["integration_batch_size"])
            self._show_progress = bool(f.attrs["show_progress"])
            self._force_energy_node_caching = bool(f.attrs["force_energy_node_caching"])
            self._reduce_memory = bool(f.attrs["reduce_memory"])
            self._n_intervals = int(f.attrs["n_intervals"])
            self._density_integration_nodes = f["density_integration_nodes"][:].tolist()
            self._energy_range = processed_energy_range(f["energy_range"][:].tolist())
            self._peak_nodes = f["peak_nodes"][:].tolist()

            if "offset" in f.attrs:
                self._offset = f.attrs["offset"]
            else:
                self._offset = None

            if "valid_events" in f.attrs:
                self._valid_events = int(f.attrs["valid_events"])
            else:
                self._valid_events = None

            if "irf_cache" in f:
                # 1. Load the flat 2D tensor and the array of original lengths
                flat_tensor = torch.from_numpy(f["irf_cache"][:])
                lengths = f["irf_cache_lengths"][:]
                
                # 2. Use torch.split to instantly reconstruct the exact ragged list
                self._irf_cache = list(torch.split(flat_tensor, list(lengths), dim=0))
            else:
                self._irf_cache = None

            if "irf_energy_node_cache" in f:
                flat_nodes = f["irf_energy_node_cache"][:]
                lengths = f["irf_energy_node_cache_lengths"][:]
                
                # Calculate split boundaries and reconstruct the original list
                split_indices = np.cumsum(lengths)[:-1]
                self._irf_energy_node_cache = np.split(flat_nodes, split_indices)
                
            else:
                self._irf_energy_node_cache = None

            if "area_cache" in f:
                self._area_cache = f["area_cache"][:]
            else:
                self._area_cache = None

            if "area_energy_node_cache" in f:
                self._area_energy_node_cache = f["area_energy_node_cache"][:]
            else:
                self._area_energy_node_cache = None

            if "exp_events" in f:
                self._exp_events = float(f["exp_events"][()])
            else:
                self._exp_events = None

            if "exp_density" in f:
                self._exp_density = torch.from_numpy(f["exp_density"][:])
            else:
                self._exp_density = None

            if "valid_mask_cache" in f:
                flat_nodes = f["valid_mask_cache"][:]
                lengths = f["valid_mask_cache_lengths"][:]
                
                # Calculate split boundaries and reconstruct the original list
                split_indices = np.cumsum(lengths)[:-1]
                self._valid_mask_cache = np.split(flat_nodes, split_indices)
                
            else:
                self._valid_mask_cache = None

            if "last_convolved_source_dict_number" in f.attrs:
                self._last_convolved_source_dict_number = json.loads(
                    f.attrs["last_convolved_source_dict_number"]
                )
            else:
                self._last_convolved_source_dict_number = None

            if "last_convolved_source_dict_density" in f.attrs:
                self._last_convolved_source_dict_density = json.loads(
                    f.attrs["last_convolved_source_dict_density"]
                )
            else:
                self._last_convolved_source_dict_density = None

            if "last_convolved_lon_deg" in f.attrs:
                lon = f.attrs["last_convolved_lon_deg"]
                lat = f.attrs["last_convolved_lat_deg"]
                frame = f.attrs["last_convolved_frame"]
                equinox = f.attrs.get("last_convolved_equinox", None)

                self._last_convolved_source_skycoord = SkyCoord(
                    lon, lat, unit="deg", frame=frame, equinox=equinox
                )
            else:
                self._last_convolved_source_skycoord = None

            if self._irf_cache is not None:
                self._init_node_pool()

    def _compute_density_for_indices(
        self, indices: np.ndarray, sc_coord_sph, earth_occ_index, coord, pix_ind
    ) -> np.ndarray:
        self._init_node_pool()

        active_pool = True
        if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
            active_pool = self._irf.active_pool
            if not active_pool:
                self._irf.init_compute_pool()

        res_block, nodes = self._compute_density_helper(
            indices,
            coord,
            sc_coord_sph=sc_coord_sph,
            earth_occ_index=earth_occ_index,
            buffer=None,
        )

        if (
            isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction)
            and not active_pool
        ):
            self._irf.shutdown_compute_pool()

        flux = torch.as_tensor(
            self._source(np.array([self._lon[pix_ind]]), np.array([self._lat[pix_ind]]), np.asarray(nodes, dtype=np.float64).ravel()),
            dtype=torch.float64,
        ).view(nodes.shape)

        exp_density = torch.zeros(len(indices), dtype=torch.float64)
        torch.linalg.vecdot(res_block.to(torch.float64), flux, dim=1, out=exp_density)

        result = np.asarray(exp_density, dtype=np.float64)
        if self._offset is not None:
            result += self._offset

        return result

    def _integrate_single_event_density(
        self,
        event_idx: int,
        occ_val: float,
        live_val: float,
        lon_val: float,
        lat_val: float,
        relerr: float,
        abserr: float,
        maxEval: int,
    ) -> tuple[float, bool]:
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

            photons = PhotonListWithDirectionAndEnergyInSCFrame(
                lon_src, lat_src, energies.astype(np.float32)
            )
            events = EmCDSEventDataInSCFrameFromArrays(
                e_meas_arr, lon_sc_arr, lat_sc_arr, phi_m_arr
            )

            diff_area = asarray(
                self._irf.differential_effective_area_cm2(photons, events),
                dtype=np.float64,
            )
            flux = np.zeros(num_eval)
            for pix_ind, pix in enumerate(self._pixels):
                flux += self._source(np.array([self._lon[pix_ind]]), np.array([self._lat[pix_ind]]),energies)

            flux = asarray(flux, dtype=np.float64)
            return diff_area * occ_val * live_val * flux

        total_res_val = 0.0
        total_err = 0.0

        for x in self._energy_range:
            if isinstance(x, list):
                res, err = cubature(
                    density_integrand,
                    ndim=1,
                    fdim=1,
                    xmin=[x[0]],
                    xmax=[x[1]],
                    vectorized=True,
                    relerr=relerr,
                    abserr=abserr,
                    maxEval=maxEval,
                )
                total_res_val += res[0]
                total_err += err[0]
            else:
                Ei = np.array([[float(x)]], dtype=np.float64)
                val = density_integrand(Ei)
                total_res_val += val[0]

        allowed_err = max(abserr, relerr * abs(total_res_val))
        return total_res_val, bool(total_err <= allowed_err)

    def _integrate_total_counts(
        self, coord, relerr: float, abserr: float, maxEval: int
    ) -> tuple[float, bool]:
        from cubature import cubature

        sc_coord_sph = self._get_target_in_sc_frame(coord, self._sc_ori_center)
        lon_ph_rad_center = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad_center = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        earth_occ_center = self._earth_occ(coord, self._sc_ori_center)
        combined_time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(
            np.float32
        ) * earth_occ_center

        def total_counts_integrand(Ei):
            energies = Ei[:, 0]
            num_eval = len(energies)
            total_counts_for_energies = np.zeros(num_eval, dtype=np.float64)

            
            for i_e, E in enumerate(energies):
                batch_energies = np.full(len(lon_ph_rad_center), E, dtype=np.float32)
                photons = PhotonListWithDirectionAndEnergyInSCFrame(
                    lon_ph_rad_center, lat_ph_rad_center, batch_energies
                )
                eff_areas = asarray(
                    self._irf.effective_area_cm2(photons), dtype=np.float64
                )

                total_area_at_E = np.sum(eff_areas * combined_time_weights)
                
                for pix_ind, pix in enumerate(self._pixels):
                    flux_at_E = self._source(np.array([self._lon[pix_ind]]), np.array([self._lat[pix_ind]]), np.array([E]))[0]
                    total_counts_for_energies[i_e] += total_area_at_E * flux_at_E

            return total_counts_for_energies

        high_prec_total_counts = 0.0
        total_err_total = 0.0

        for x in self._energy_range:
            if isinstance(x, list):
                res_total, err_total = cubature(
                    total_counts_integrand,
                    ndim=1,
                    fdim=1,
                    xmin=[x[0]],
                    xmax=[x[1]],
                    vectorized=True,
                    relerr=relerr,
                    abserr=abserr,
                    maxEval=maxEval,
                )
                high_prec_total_counts += float(res_total[0])
                total_err_total += float(err_total[0])
            else:
                Ei = np.array([[float(x)]], dtype=np.float64)
                val_total = total_counts_integrand(Ei)
                high_prec_total_counts += float(val_total[0])

        allowed_err_total = max(abserr, relerr * abs(high_prec_total_counts))
        return high_prec_total_counts, bool(total_err_total <= allowed_err_total)

    def validate_integration(
        self,
        n_events: int,
        relerr: float = 2e-4,
        abserr: float = 1e-30,
        maxEval: int = 1000000,
        save_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        pool_was_active = True
        if isinstance(self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction):
            pool_was_active = self._irf.active_pool
            if not pool_was_active:
                self._irf.init_compute_pool()

        try:

            if (self._area_cache is None):
                self._compute_area()

            
            high_prec_densities = []
            converged_mask = []
            high_prec_total_counts = 0
            opt_densities = np.array([])
            opt_total_counts = 0
            
            for pix_ind, pix in enumerate(self._pixels):
                
                sc_coord_sph = self._sc_coord_sph_cache[pix_ind]
                self._valid_events = int(len(self._valid_mask_cache[pix_ind]))
                earth_occ_index = self._earth_occ(self._source_coord_cache[pix_ind], self._sc_ori_unique)[self._inv_idx]
                
                sampled_indices = np.random.choice(
                    self._valid_mask_cache[pix_ind],
                    size=min(n_events, self._valid_events),
                    replace=False,
                )
    
                sc_coord_sph_sampled = sc_coord_sph[sampled_indices]
                earth_occ_sampled = earth_occ_index[sampled_indices]
                livetime_ratio_sampled = self._livetime_ratio[sampled_indices]

                #compute Total count
                high_prec_pixel_counts, pixel_counts_converged = (
                    self._integrate_total_counts(self._source_coord_cache[pix_ind], relerr, abserr, maxEval)
                )

                if not pixel_counts_converged:
                    raise RuntimeError(
                    f"High-precision total counts integration did not converge for the pixel {pix}. Try increasing the maximum number of evaluations or the acceptable error."
                )

                high_prec_total_counts += high_prec_pixel_counts

                #Compute opt densities
                opt_densities_pixel = self._compute_density_for_indices(
                    sampled_indices, sc_coord_sph, earth_occ_index, self._source_coord_cache[pix_ind], pix_ind
                )

                #concatenate the opt densities
                opt_densities = np.concatenate( (opt_densities,opt_densities_pixel), axis= 0 )
                
                for idx, event_idx in enumerate(
                    tqdm(
                        sampled_indices,
                        desc="Computing high-precision densities",
                        disable=not self.show_progress,
                    )
                ):
                    res_val, is_converged = self._integrate_single_event_density(
                        event_idx,
                        earth_occ_sampled[idx],
                        livetime_ratio_sampled[idx],
                        sc_coord_sph_sampled[idx].lon.rad,
                        sc_coord_sph_sampled[idx].lat.rad,
                        relerr,
                        abserr,
                        maxEval,
                    )
                    if self._offset is not None:
                        res_val += self._offset
                    high_prec_densities.append(res_val)
                    converged_mask.append(is_converged)
            
                flux_area = self._source(np.array([self._lon[pix_ind]]), np.array([self._lat[pix_ind]]), self._area_energy_node_cache)
                opt_total_counts += float(np.sum(self._area_cache[pix_ind] * flux_area, dtype=float))

            
            high_prec_densities = np.array(high_prec_densities, dtype=np.float64)
            converged_mask = np.array(converged_mask, dtype=bool)

            if sum(converged_mask) / len(converged_mask) < 0.9:
                raise RuntimeError(
                    "Less than 90% of high-precision densities converged. Try increasing the maximum number of evaluations or the acceptable error."
                )

            

            
            
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_errors = (opt_densities / high_prec_densities) - 1.0
                rel_errors = np.nan_to_num(rel_errors, nan=0.0, posinf=0.0, neginf=0.0)

            rel_deviation_total = (opt_total_counts / high_prec_total_counts) - 1.0
            converged_rel_errors = rel_errors[converged_mask]

            if len(converged_rel_errors) > 0:
                mean_err = float(np.mean(converged_rel_errors))
                median_err = float(np.median(converged_rel_errors))
                std_err = float(np.std(converged_rel_errors, ddof=1))
            else:
                mean_err = median_err = std_err = float("nan")

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
                },
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
                        "raw_relative_errors": results["density_errors"][
                            "raw_relative_errors"
                        ].tolist(),
                        "sampled_indices": results["density_errors"][
                            "sampled_indices"
                        ].tolist(),
                        "converged_mask": results["density_errors"][
                            "converged_mask"
                        ].tolist(),
                    },
                }
                with open(str(save_path), "w") as f:
                    json.dump(serializable_results, f, indent=4)

            return results

        finally:
            if not pool_was_active and isinstance(
                self._irf, UnpolarizedNFFarFieldInstrumentResponseFunction
            ):
                self._irf.shutdown_compute_pool()

    @staticmethod
    def report_and_plot_validation(
        validation_results: Union[dict, str, Path],
        save_path: Optional[Union[str, Path]] = None,
    ):
        if isinstance(validation_results, (str, Path)):
            if not os.path.exists(str(validation_results)):
                raise FileNotFoundError(
                    f"Validation file {str(validation_results)} not found."
                )
            with open(str(validation_results), "r") as f:
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
        print(
            f"  Convergence:  {de['num_converged']/de['total_sampled']*100:.2f}% of all sampled events converged"
        )
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
        plot_range = (
            (p1, p99) if p1 < p99 else (converged_errors.min(), converged_errors.max())
        )

        ax.hist(
            converged_errors,
            bins="fd",
            range=plot_range,
            alpha=0.75,
            color="royalblue",
            edgecolor="black",
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Zero Error")
        ax.axvline(
            de["median"] * 100,
            color="darkorange",
            linestyle="-",
            label=f"Median ({de['median']*100:.1e}%)",
        )
        ax.set_xlabel("Relative Error (%)", fontsize=11)

        ax.set_title(
            "Distribution of Relative Errors in Expectation Densities",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Number of Events", fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(str(save_path), bbox_inches="tight", dpi=300)

        plt.show()

    def expected_counts(self) -> float:
        """
        Return the total expected counts.
        """
        self._update_cache()
        source_dict = self._source.to_dict()

        if (source_dict != self._last_convolved_source_dict_number) or (
            self._exp_events is None
        ):
            self._exp_events =  0
            for pix_ind, pix in enumerate(self._pixels):
                
                area = self._area_cache[pix_ind]
                flux = self._source(np.array([self._lon[pix_ind]]), np.array([self._lat[pix_ind]]), self._area_energy_node_cache)
                self._exp_events += np.sum(area * flux, dtype=float)
                #print(self._exp_events)
                #print(area)
                #print(flux)

        self._last_convolved_source_dict_number = source_dict
        return self._exp_events

    def expectation_density(self) -> Iterable[float]:
        """
        Return the expected number of counts density. This equals the event probabiliy times the number of events.
        """

        self._update_cache()
        source_dict = self._source.to_dict()
        if (source_dict != self._last_convolved_source_dict_density) or (
            self._exp_density is None
        ):
            self._exp_density = torch.zeros(self._n_events, dtype=torch.float64)

            n_energy = self.total_density_integration_nodes
            batch_size = self._integration_batch_size // n_energy

            for pix in range(len(self._pixels)):
            
                self._valid_events = int(len(self._valid_mask_cache[pix]))
    
                if (self._irf_energy_node_cache[pix] is not None) & (
                    batch_size >= self._valid_events
                ):
                    flux = torch.as_tensor(
                        self._source( np.array([self._lon[pix]]), np.array([self._lat[pix]]),
                            np.asarray(
                                self._irf_energy_node_cache[pix], dtype=np.float64
                            ).ravel()
                        ),
                        dtype=torch.float64,
                    ).view(self._irf_energy_node_cache[pix].shape)
    
                    cache = torch.as_tensor(self._irf_cache[pix], dtype=torch.float64)
    
                    self._exp_density[self._valid_mask_cache[pix]] += torch.linalg.vecdot(
                        cache, flux, dim=1
                    )
    
                else:
                    if self._irf_energy_node_cache[pix] is None:
                        sc_coord_sph = self._sc_coord_sph_cache[pix][self._valid_mask_cache[pix]]
    
                        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
                        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
    
                        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(
                            torch.as_tensor(lon_ph_rad),
                            torch.as_tensor(lat_ph_rad),
                            indices=self._valid_mask_cache[pix],
                        )
    
                    for i in range(0, self._valid_events, batch_size):
                        end = min(i + batch_size, self._valid_events)
    
                        if self._irf_energy_node_cache[pix] is None:
                            e_sl = self._energy_m_keV[self._valid_mask_cache[pix][i:end]]
                            p_sl = self._phi_rad[self._valid_mask_cache[pix][i:end]]
                            pg_sl = phi_geo_rad[i:end]
                            pig_sl = phi_igeo_rad[i:end]
    
                            nodes, _ = self._get_nodes(e_sl, p_sl, pg_sl, pig_sl)
                        else:
                            nodes = self._irf_energy_node_cache[pix][i:end]
    
                        nodes = np.asarray(nodes, dtype=np.float64)
    
                        flux_batch = torch.as_tensor(
                            self._source(np.array([self._lon[pix]]), np.array([self._lat[pix]]), nodes.ravel()), dtype=torch.float64
                        ).view(nodes.shape)
    
                        cache = torch.as_tensor(self._irf_cache[pix][i:end], dtype=torch.float64)
    
                        self._exp_density[self._valid_mask_cache[pix][i:end]] += (
                            torch.linalg.vecdot(cache, flux_batch, dim=1)
                        )

        self._last_convolved_source_dict_density = source_dict

        result = np.asarray(self._exp_density, dtype=np.float64)

        if self._offset is not None:
            return result + self._offset
        else:
            return result


    def _set_mask(self):
        """
        Mask the pixels, True means we mask and False we keep
        """
        
        if len(self._mask) != self._npix :
            raise ValueError(f"The dimension of your mask array ({len(self._mask)}) should match the number of pixel ({self._npix})")

        listofpixels = np.arange(self._npix)
        
        return listofpixels[~self._mask]   
