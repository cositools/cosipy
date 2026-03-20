import numpy as np
from typing import Iterable

from cosipy.interfaces.expectation_interface import ExpectationDensityInterface
from cosipy.statistics.likelihood_functions import UnbinnedLikelihood

class MockExpectationDensity(ExpectationDensityInterface):
    """
    A minimal mock implementation of ExpectationDensityInterface 
    to feed predictable data into UnbinnedLikelihood for testing.
    """
    def __init__(self, counts: float, density: Iterable[float]):
        self._counts = counts
        self._density = density

    def expected_counts(self) -> float:
        return self._counts

    def expectation_density(self) -> Iterable[float]:
        return self._density


def test_unbinned_likelihood_nobservations():
    """Test that the nobservations property correctly counts the density iterable."""
    mock_exp = MockExpectationDensity(counts=10.0, density=[2.0, 3.0, 5.0])
    likelihood = UnbinnedLikelihood(expectation=mock_exp)
    
    assert likelihood.nobservations == 3


def test_unbinned_likelihood_get_log_like_success():
    """Test the correct mathematical calculation of the log-likelihood."""

    counts = 10.0
    density = [2.0, 3.0, 5.0]
    mock_exp = MockExpectationDensity(counts=counts, density=density)
    
    likelihood = UnbinnedLikelihood(expectation=mock_exp)
    result = likelihood.get_log_like()
    
    expected_log_like = np.log(30.0) - counts
    
    assert np.isclose(result, expected_log_like)
    assert likelihood.nobservations == 3


def test_unbinned_likelihood_negative_or_zero_density():
    """Test that a density <= 0 immediately returns -np.inf."""

    mock_exp = MockExpectationDensity(counts=10.0, density=[2.0, 0.0, 5.0])
    likelihood = UnbinnedLikelihood(expectation=mock_exp)
    
    result = likelihood.get_log_like()
    
    assert result == -np.inf


def test_unbinned_likelihood_with_generator_and_batching():
    """Test that batching works correctly when given a generator instead of a list."""
    
    def density_generator():
        for val in [2.0, 3.0, 5.0, 4.0]:
            yield val

    counts = 15.0
    mock_exp = MockExpectationDensity(counts=counts, density=density_generator())
    
    likelihood = UnbinnedLikelihood(expectation=mock_exp, batch_size=2)
    result = likelihood.get_log_like()
    
    expected_log_like = np.log(120.0) - counts
    
    assert np.isclose(result, expected_log_like)
    assert likelihood.nobservations == 4