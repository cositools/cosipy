"""
Integration tests for image deconvolution algorithms.
Each class groups tests for one algorithm. Within a class, tests share
a common parameter template via a fixture, and each test method exercises
a specific scenario (e.g. with/without background normalization, different
stopping criteria). Numerical values are regression checks: they should
only be updated intentionally when the algorithm changes.
"""

import pytest
import numpy as np
from yayc import Configurator

from cosipy.image_deconvolution import (
    RichardsonLucyBasic,
    RichardsonLucy,
    RichardsonLucyAdvanced,
    MAP_RichardsonLucy,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(algorithm, num_iteration):
    """Initialize and iterate an algorithm. Returns the algorithm instance."""
    algorithm.initialization()
    for _ in range(num_iteration):
        if algorithm.iteration():
            break
    algorithm.finalization()
    return algorithm


# ---------------------------------------------------------------------------
# RichardsonLucyBasic
# ---------------------------------------------------------------------------

class TestRichardsonLucyBasic:

    @pytest.fixture
    def parameter(self, tmp_path):
        return Configurator({
            "iteration_max": 3,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_basic(self, dataset, model, mask, parameter):
        alg = RichardsonLucyBasic(model, dataset, mask, parameter)
        run(alg, parameter["iteration_max"])


# ---------------------------------------------------------------------------
# RichardsonLucy
# ---------------------------------------------------------------------------

class TestRichardsonLucy:

    @pytest.fixture
    def parameter(self, tmp_path):
        return Configurator({
            "iteration_max": 3,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_basic(self, dataset, model, mask, parameter):
        alg = RichardsonLucy(model, dataset, mask, parameter)
        run(alg, parameter["iteration_max"])


# ---------------------------------------------------------------------------
# RichardsonLucyAdvanced
# ---------------------------------------------------------------------------

class TestRichardsonLucyAdvanced:
    """
    Tests for RichardsonLucyAdvanced covering acceleration on/off.
    Numerical assertions are regression checks.
    """

    NUM_ITERATION = 3

    @pytest.fixture
    def base_parameter(self, tmp_path):
        return Configurator({
            "iteration_max": self.NUM_ITERATION,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "acceleration": {"activate": True, "alpha_max": 10.0},
            "response_weighting": {"activate": True, "index": 0.5},
            "smoothing": {"activate": True, "FWHM": {"value": 2.0, "unit": "deg"}},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}},
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_with_acceleration(self, dataset, model, mask, base_parameter):
        alg = run(RichardsonLucyAdvanced(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-likelihood'][0], 5495.120521335304)

    def test_without_acceleration(self, dataset, model, mask, base_parameter):
        base_parameter["acceleration:activate"] = False
        alg = run(RichardsonLucyAdvanced(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-likelihood'][0], 5270.562770130176)


# ---------------------------------------------------------------------------
# MAP_RichardsonLucy
# ---------------------------------------------------------------------------

class TestMAPRichardsonLucy:
    """
    Tests for MAP_RichardsonLucy covering different prior configurations
    and stopping criteria. Numerical assertions are regression checks.
    """

    NUM_ITERATION = 10

    @pytest.fixture
    def base_parameter(self, tmp_path):
        return Configurator({
            "iteration_max": self.NUM_ITERATION,
            "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
            "response_weighting": {"activate": True, "index": 0.5},
            "background_normalization_optimization": {"activate": True, "range": {"bkg": [0.9, 1.1]}},
            "stopping_criteria": {"statistics": "log-posterior", "threshold": 1e-2},
            "prior": {
                "TSV": {"coefficient": 1e-10},
                "gamma": {
                    "model":      {"theta": {"value": np.inf, "unit": "cm-2 s-1 sr-1"}, "k": {"value": 0.999}},
                    "background": {"theta": {"value": np.inf}, "k": {"value": 1.0}},
                },
            },
            "save_results": {"activate": True, "directory": str(tmp_path), "only_final_result": True},
        })

    def test_with_gamma_and_tsv_prior(self, dataset, model, mask, base_parameter):
        """Full setup: TSV + gamma prior, bkg optimization enabled."""
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-posterior'], 6567.857548203495)

    def test_without_bkg_optimization(self, dataset, model, mask, base_parameter):
        """Background normalization fixed at 1.0."""
        base_parameter["background_normalization_optimization:activate"] = False
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-posterior'], 6202.336733778631)

    def test_stopping_criteria_threshold(self, dataset, model, mask, base_parameter):
        """Large threshold causes early stopping after 2 iterations."""
        base_parameter["stopping_criteria:threshold"] = 1e10
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        assert len(alg.results) == 2

    def test_stopping_criteria_log_likelihood(self, dataset, model, mask, base_parameter):
        """Use log-likelihood instead of log-posterior as stopping criterion."""
        base_parameter["background_normalization_optimization:activate"] = False
        base_parameter["stopping_criteria:threshold"] = 1e10
        base_parameter["stopping_criteria:statistics"] = "log-likelihood"
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-likelihood'][0], 3931.3528198012773)

    def test_without_gamma_prior(self, dataset, model, mask, base_parameter):
        """TSV prior only, no gamma prior."""
        base_parameter["background_normalization_optimization:activate"] = False
        base_parameter["stopping_criteria:statistics"] = "log-posterior"
        base_parameter["stopping_criteria:threshold"] = 1e10
        base_parameter["prior"] = {"TSV": {"coefficient": 1e-10}}
        alg = run(MAP_RichardsonLucy(model, dataset, mask, base_parameter), self.NUM_ITERATION)
        # Regression check
        assert np.isclose(alg.results[-1]['log-posterior'], 3931.29811442966)

    def test_invalid_stopping_statistics_raises(self, dataset, model, mask, base_parameter):
        """Invalid stopping_criteria:statistics should raise ValueError."""
        base_parameter["stopping_criteria:statistics"] = "likelihooooooooood!!!"
        with pytest.raises(ValueError):
            MAP_RichardsonLucy(model, dataset, mask, base_parameter)
