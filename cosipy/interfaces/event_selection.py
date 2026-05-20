from typing import Protocol, runtime_checkable, Iterable, Union, Type

from . import EventInterface, EventDataInterface
from .data_interface import is_single_event

@runtime_checkable
class EventSelectorInterface(Protocol):

    event_data_type = EventDataInterface

    @property
    def event_type(self) -> Type[EventInterface]:
        return self.event_data_type.event_type

    def select(self, events:Union[EventInterface, EventDataInterface],
               early_stop:bool = True) -> Union[bool, Iterable[bool]]:
        """
        True to keep an event

        Parameters
        ----------
        EventDataInterface:

        early_stop: If True (default), the implementation might raise
        a StopIteration condition if all subsequent event will yield
        select=False. If False, the implementation will continue to
        yield select=False such that the size of the output matches
        the number of input events.

        Return a single value for a single Event.
        As many values as events for EventData

        Implementation can define only _select assuming multiple
        events, and let this default function handle single event case

        """

        single_event = is_single_event(events)

        if single_event:
            events = self.event_data_type.from_event(events)
            return next(iter(self._select(events, early_stop = False)))
        else:
            return self._select(events, early_stop)

        def _select(self, events:EventDataInterface,
                    early_stop:bool = True) -> Iterable[bool]:
            """
            This allows implementation to only define the behaviour for list,
            and let the above function handle the case of single event.

            """
