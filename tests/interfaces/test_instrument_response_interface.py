from cosipy.interfaces.instrument_response_interface import FarFieldInstrumentResponseFunctionInterface


class _DummyIRF(FarFieldInstrumentResponseFunctionInterface):
    def _differential_effective_area_cm2(self, photons, events):
        return iter([1.0, 0.0, 2.0])

    def _effective_area_cm2(self, photons):
        return iter([2.0, 0.0, 0.0])


def test_default_event_probability_is_zero_where_effective_area_is_zero():
    probs = _DummyIRF()._event_probability(None, None)

    # Lazy: no per-element work until iterated
    assert iter(probs) is probs

    assert list(probs) == [0.5, 0.0, 0.0]
