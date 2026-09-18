from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from cosipy.interfaces.data_interface import EventDataWithEnergyInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import asarray


class EnergySelector(EventSelectorInterface):

    event_data_type = EventDataWithEnergyInterface

    def __init__(self, energy_ranges: Quantity = None):
        """
        Selects events whose measured energy falls within any of the
        given [min, max) ranges.

        Parameters
        ----------
        energy_ranges: Quantity of shape (N, 2) or (2,)
            N (or a single) [min, max) energy range(s), inclusive
            minimum / exclusive maximum. Ranges may overlap or be given
            out of order; they are canonicalized (sorted,
            overlapping/adjacent ranges merged, empty/invalid ranges
            dropped) on construction. Default: None -> a single
            [0, inf) keV range (no cut).
        """

        if energy_ranges is None:
            energy_ranges = Quantity([[0., np.inf]], u.keV)

        arr = np.asarray(Quantity(energy_ranges).to_value(u.keV), dtype=float)

        if arr.ndim == 1:
            if arr.shape != (2,):
                raise ValueError("energy_ranges must have shape (N, 2) or (2,)")
            arr = arr[None, :]
        elif arr.ndim != 2 or arr.shape[-1] != 2:
            raise ValueError("energy_ranges must have shape (N, 2) or (2,)")

        self._energy_ranges_keV = self._merge_ranges(arr)

    @staticmethod
    def _merge_ranges(ranges_keV: np.ndarray) -> np.ndarray:
        """
        Sort, drop empty/invalid (hi <= lo), and merge overlapping or
        touching [min, max) ranges into minimal canonical form.
        """

        ranges_keV = ranges_keV[ranges_keV[:, 1] > ranges_keV[:, 0]]

        if len(ranges_keV) == 0:
            return ranges_keV.reshape(0, 2)

        sorted_ranges = ranges_keV[np.argsort(ranges_keV[:, 0])]

        merged = [sorted_ranges[0].copy()]
        for lo, hi in sorted_ranges[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append(np.array([lo, hi]))

        return np.array(merged)

    @property
    def energy_ranges_keV(self) -> np.ndarray:
        """(N, 2) array of [min, max) ranges, in keV, as plain floats."""
        return self._energy_ranges_keV.copy()

    @property
    def energy_ranges(self) -> Quantity:
        """(N, 2) Quantity of [min, max) ranges."""
        return Quantity(self._energy_ranges_keV, u.keV)

    def union(self, other: "EnergySelector") -> "EnergySelector":
        """Ranges selected by either self or other."""

        if not isinstance(other, EnergySelector):
            raise TypeError(f"union() expects another EnergySelector, got {type(other)}")

        combined = np.concatenate([self._energy_ranges_keV, other._energy_ranges_keV], axis=0)

        return EnergySelector(Quantity(combined, u.keV))

    def intersect(self, other: "EnergySelector") -> "EnergySelector":
        """Ranges selected by both self and other."""

        if not isinstance(other, EnergySelector):
            raise TypeError(f"intersect() expects another EnergySelector, got {type(other)}")

        los = np.maximum(self._energy_ranges_keV[:, None, 0], other._energy_ranges_keV[None, :, 0])
        his = np.minimum(self._energy_ranges_keV[:, None, 1], other._energy_ranges_keV[None, :, 1])

        combined = np.stack([los.ravel(), his.ravel()], axis=1)

        return EnergySelector(Quantity(combined, u.keV))  # constructor drops empty (his<=los) pairs

    def except_(self, other: "EnergySelector") -> "EnergySelector":
        """Ranges selected by self but not other (set difference)."""

        if not isinstance(other, EnergySelector):
            raise TypeError(f"except_() expects another EnergySelector, got {type(other)}")

        result = []
        for a, b in self._energy_ranges_keV:
            cur = a
            for lo, hi in other._energy_ranges_keV:  # sorted, non-overlapping
                if hi <= cur or lo >= b:
                    continue
                if lo > cur:
                    result.append([cur, min(lo, b)])
                cur = max(cur, hi)
                if cur >= b:
                    break
            if cur < b:
                result.append([cur, b])

        combined = np.array(result) if result else np.empty((0, 2))

        return EnergySelector(Quantity(combined, u.keV))

    def _select(self, events: EventDataWithEnergyInterface,
                early_stop: bool = True) -> Iterable[bool]:

        energy_keV = np.asarray(asarray(events.energy_keV, dtype=np.float64, force_dtype=False))
        lo, hi = self._energy_ranges_keV[:, 0], self._energy_ranges_keV[:, 1]

        return np.any((energy_keV[..., None] >= lo) & (energy_keV[..., None] < hi), axis=-1)
