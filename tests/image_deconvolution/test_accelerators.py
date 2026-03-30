import pytest
import numpy as np

from cosipy.image_deconvolution.algorithms.accelerators.accelerator_base import EMStepResult
from cosipy.image_deconvolution.algorithms.accelerators.max_step_accelerator import MaxStepAccelerator
from cosipy.image_deconvolution.algorithms.accelerators.line_search_accelerator import LineSearchAccelerator

def make_model(dataset, value=1.0):
    from cosipy.image_deconvolution import AllSkyImageModel
    model = AllSkyImageModel(
        dataset[0].model_axes['lb'].nside,
        dataset[0].model_axes['Ei'].edges,
    )
    model[:] = value * model.unit
    return model


def make_em_results(dataset, model_value=1.0, delta_value=0.1, bkg_norm_value=1.0):
    """Return a (before, after) pair of EMStepResult."""
    model_before = make_model(dataset, value=model_value)
    model_after  = make_model(dataset, value=model_value + delta_value)

    bkg_before = {"bkg": bkg_norm_value}
    bkg_after  = {"bkg": bkg_norm_value + 0.01}

    src_exp_before = dataset.calc_source_expectation_list(model_before)
    src_exp_after  = dataset.calc_source_expectation_list(model_after)

    before = EMStepResult(
        model                   = model_before,
        dict_bkg_norm           = bkg_before,
        expectation_list        = dataset.combine_expectation_list(src_exp_before, dataset.calc_bkg_expectation_list(bkg_before)),
        source_expectation_list = src_exp_before,
        bkg_expectation_list    = dataset.calc_bkg_expectation_list(bkg_before),
    )
    after = EMStepResult(
        model                   = model_after,
        dict_bkg_norm           = bkg_after,
        expectation_list        = dataset.combine_expectation_list(src_exp_after, dataset.calc_bkg_expectation_list(bkg_after)),
        source_expectation_list = src_exp_after,
        bkg_expectation_list    = dataset.calc_bkg_expectation_list(bkg_after),
    )
    return [before, after]


def check_result(result, extra_keys):
    """Common assertions for AcceleratorResult."""
    assert np.all(result.model.contents >= 0), "model must be non-negative"
    for key in extra_keys:
        assert key in result.extras, f"extras must contain '{key}'"
        if key in ["accel_factor", "accel_factor_bkg"]:
            assert result.extras[key] >= 1.0, f"{key} must be >= 1.0"


# ---------------------------------------------------------------------------
# MaxStepAccelerator
# ---------------------------------------------------------------------------

class TestMaxStepAccelerator:

    @pytest.fixture
    def accelerator(self):
        from yayc import Configurator
        return MaxStepAccelerator(Configurator({"accel_factor_max": 10.0, "accel_bkg_norm": False}))

    def test_basic(self, accelerator, dataset, mask):
        result = accelerator.compute(make_em_results(dataset), dataset, mask)
        print(f"accel_factor = {result.extras['accel_factor']}")
        check_result(result, ["accel_factor"])

    def test_with_bkg_norm(self, dataset, mask):
        from yayc import Configurator
        acc = MaxStepAccelerator(Configurator({"accel_factor_max": 10.0, "accel_bkg_norm": True}))
        result = acc.compute(make_em_results(dataset), dataset, mask)
        print(f"accel_factor = {result.extras['accel_factor']}")
        check_result(result, ["accel_factor"])


# ---------------------------------------------------------------------------
# LineSearchAccelerator
# ---------------------------------------------------------------------------

class TestLineSearchAccelerator:

    def make_accelerator(self, params):
        from yayc import Configurator
        return LineSearchAccelerator(Configurator(params))

    def test_1d_no_bkg(self, dataset, mask):
        """Model only, 1-D line search."""
        acc = self.make_accelerator({"accel_factor_max": 10.0})
        result = acc.compute(make_em_results(dataset), dataset, mask)
        print(f"accel_factor = {result.extras['accel_factor']}")
        check_result(result, ["accel_factor", "accel_factor_bkg"])

    def test_1d_with_bkg_same_factor(self, dataset, mask):
        """Model and bkg scaled by the same accel_factor."""
        acc = self.make_accelerator({
            "accel_factor_max": 10.0,
            "accel_bkg_norm": {"activate": True, "independent": False, "max": 10.0},
        })
        result = acc.compute(make_em_results(dataset), dataset, mask)
        print(f"accel_factor = {result.extras['accel_factor']}, accel_factor_bkg = {result.extras['accel_factor_bkg']}")
        check_result(result, ["accel_factor", "accel_factor_bkg"])
        assert result.extras["accel_factor"] == pytest.approx(result.extras["accel_factor_bkg"])

    def test_2d_independent_gradient(self, dataset, mask):
        """Independent bkg accel_factor, gradient search."""
        acc = self.make_accelerator({
            "accel_factor_max": 10.0,
            "accel_bkg_norm": {"activate": True, "independent": True, "max": 10.0,
                               "search_method": "gradient", "grid_n": 5},
        })
        result = acc.compute(make_em_results(dataset), dataset, mask)
        print(f"accel_factor = {result.extras['accel_factor']}, accel_factor_bkg = {result.extras['accel_factor_bkg']}")
        check_result(result, ["accel_factor", "accel_factor_bkg"])

    def test_2d_independent_grid(self, dataset, mask):
        """Independent bkg accel_factor, grid search."""
        acc = self.make_accelerator({
            "accel_factor_max": 10.0,
            "accel_bkg_norm": {"activate": True, "independent": True, "max": 10.0,
                               "search_method": "grid", "grid_n": 5},
        })
        result = acc.compute(make_em_results(dataset), dataset, mask)
        print(f"accel_factor = {result.extras['accel_factor']}, accel_factor_bkg = {result.extras['accel_factor_bkg']}")
        check_result(result, ["accel_factor", "accel_factor_bkg"])

    def test_invalid_search_method_raises(self):
        from yayc import Configurator
        with pytest.raises(ValueError):
            LineSearchAccelerator(Configurator({
                "accel_factor_max": 10.0,
                "accel_bkg_norm": {"activate": True, "independent": True, "max": 10.0,
                                   "search_method": "invalid"},
            }))
