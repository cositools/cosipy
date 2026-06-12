import logging
logger = logging.getLogger(__name__)

import itertools
from typing import Union, Iterable

import numpy as np
from astropy.time import Time

from cosipy.interfaces import TimeTagEventInterface, EventInterface, TimeTagEventDataInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import itertools_batched, asarray


class TimeSelector(EventSelectorInterface):

    def __init__(self, tstart:Time = None, tstop:Time = None, batch_size:int = None):
        """
        Assumes events are time-ordered
        
        Selects events that fall within ANY of the time intervals defined by
        corresponding pairs of (tstart, tstop).

        Valid combinations:
        - (None, None): No time constraints
        - (Scalar, None): Single lower bound only
        - (None, Scalar): Single upper bound only  
        - (Scalar, Scalar): Single time interval
        - (List, List): Multiple time intervals (same length required)

        Parameters
        ----------
        tstart: Time, scalar Time, or None
            Start time(s) [inclusive]. If list, tstop must also be a list of same length.
        tstop: Time, scalar Time, or None
            Stop time(s) [exclusive]. If list, tstart must also be a list of same length.
        batch_size: int, default None
            Number of events to process at once
            If None, all values are processed in a single batch.
            This parameter only affects iteration when the expectation density
            is provided as an iterator that is not a numpy array. If it is already
            an array batching is not applied.
        """
        if tstart is not None and tstop is not None:
            if not tstart.isscalar == tstop.isscalar:
                logger.error("tstart and tstop must both be scalar or both be list.")
                raise ValueError

        elif tstart is None and tstop is not None:
            if tstop.isscalar == False:
                logger.error("When tstart is None, tstop must not be a list.")
                raise ValueError

        elif tstart is not None and tstop is None:
            if tstart.isscalar == False:
                logger.error("When tstop is None, tstart must not be a list.")
                raise ValueError
        
        # tstart is None and tstop is None -> OK.
        
        # Convert scalars to lists for uniform processing
        if tstart is not None and tstart.isscalar == True:
            tstart = Time([tstart])

        if tstop is not None and tstop.isscalar == True:
            tstop = Time([tstop])
        
        # length check
        if tstart is not None and tstop is not None:
            if len(tstart) != len(tstop):
                logger.error(f"tstart and tstop must have same length.")
                raise ValueError

        # Convert to relative time with respect to earlier time
        # It's faster to compare a single number than an astropy Time objects,
        # at the expense of loosing sub-ns precision for year-long durations
        # e.g. us precision for 10 years
        if tstart is not None:
            self._t0 = tstart[0]
        elif tstop is not None:
            self._t0 = tstop[0]
        else:
            self._t0 = None

        if self._t0 is None:
            self._tstart_list = None
            self._tstop_list = None
        else:
            self._tstart_list = (tstart - self._t0).jd if tstart is not None else None
            self._tstop_list = (tstop - self._t0).jd if tstop is not None else None

        self._batch_size = batch_size

    @classmethod
    def from_gti(cls, gti, batch_size:int = 10000):
        """
        Instantiate a multi time selector from good time intervals.

        Parameters
        ----------
        gti: 
            Good time intervals object with tstart_list and tstop_list attributes
        batch_size: int
            Number of events to process at once
        """
        tstart_list = gti.tstart_list
        tstop_list = gti.tstop_list

        selector = cls(tstart_list, tstop_list, batch_size)

        return selector

    def _select(self, events:TimeTagEventDataInterface, early_stop:bool = True) -> Iterable[bool]:

        def process_chunk(jd1: np.ndarray, jd2:np.ndarray):

            if self._t0 is None:
                # Means tstart=tstop=None
                return np.ones_like(jd1, dtype=bool), True

            # Relative time to t0
            time = Time(jd1, jd2, format = 'jd', copy = False)
            time = (time - self._t0).jd

            if self._tstart_list is None:
                result = time < self._tstop_list[0]

            elif self._tstop_list is None:
                result = time >= self._tstart_list[0]

            else:

                result = np.zeros(len(time), dtype=bool)

                lo_idx,hi_idx = np.searchsorted(time, [self._tstart_list, self._tstop_list], side='left')

                for lo,hi in zip(lo_idx,hi_idx):
                    result[lo:hi] = True

            # Stop further loading of event
            stop = early_stop and (self._tstop_list is not None and len(time) > 0) and time[-1] > self._tstop_list[-1]

            return result, stop

        def process_in_chunks(events):

            for chunk in itertools_batched(events, self._batch_size):

                jd1 = []
                jd2 = []

                for event in chunk:
                    jd1.append(event.jd1)
                    jd2.append(event.jd2)

                # Cache in memory
                jd1 = asarray(jd1, dtype=np.float64, force_dtype=False)
                jd2 = asarray(jd2, dtype=np.float64, force_dtype=False)

                result, stop = process_chunk(jd1, jd2)

                yield from result

                if stop:
                    return

        if (self._batch_size is None) or (isinstance(events.jd1, np.ndarray) and isinstance(events.jd2, np.ndarray)):
            results, _ = process_chunk(events.jd1, events.jd2)
            return results
        else:
            return process_in_chunks(events)



