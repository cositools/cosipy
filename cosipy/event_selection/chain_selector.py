import itertools
from typing import Iterable

from cosipy.interfaces import EventDataInterface
from cosipy.interfaces.event_selection import EventSelectorInterface


class ChainEventSelectors(EventSelectorInterface):

    def __init__(self, *selectors:EventSelectorInterface):
        self._selectors = selectors

    def _select(self, events:EventDataInterface, early_stop:bool = True) -> Iterable[bool]:

        # This is a simple implementation. It evaluates all selector and would buffer
        # (through tee) all events that are cached/batched by a given selector.
        # It should be possible to optimize this for other cases, considering:
        # - Evaluating some selectors might be costly, so we should pass only the events
        #   the survived the previous cuts
        # - The output of all selectors might be numpy arrays, so we can use masking instead

        events_copies = itertools.tee(events, len(self._selectors))
        masks = [sel._select(sel.event_data_type.fromiter(events_i), early_stop)
                 for sel, events_i in zip(self._selectors, events_copies)]

        for selected in zip(*masks):
            yield all(selected)
