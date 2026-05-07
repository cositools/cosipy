from typing import Protocol, runtime_checkable, Union
from astromodels import Model
from astromodels.sources import Source
from pathlib import Path

from .expectation_interface import BinnedExpectationInterface, ExpectationDensityInterface

from cosipy.spacecraftfile import SpacecraftHistory

__all__ = ["ThreeMLModelFoldingInterface",
           "UnbinnedThreeMLModelFoldingInterface",
           "BinnedThreeMLModelFoldingInterface",
           "ThreeMLSourceResponseInterface",
           "UnbinnedThreeMLSourceResponseInterface",
           "BinnedThreeMLSourceResponseInterface"]

@runtime_checkable
class ThreeMLModelFoldingInterface(Protocol):
    def set_model(self, model: Model):
        """
        The model is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """

@runtime_checkable
class UnbinnedThreeMLModelFoldingInterface(ThreeMLModelFoldingInterface, ExpectationDensityInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class BinnedThreeMLModelFoldingInterface(ThreeMLModelFoldingInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class ThreeMLSourceResponseInterface(Protocol):

    def set_source(self, source: Source):
        """
        The source is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
    def copy(self) -> "ThreeMLSourceResponseInterface":
        """
        This method is used to re-use the same object for multiple
        sources.
        It is expected to return a safe copy of itself
        such that when
        a new source is set, the expectation calculation
        are independent.

        psr1 = ThreeMLSourceResponse()
        psr2 = psr.copy()
        psr1.set_source(source1)
        psr2.set_source(source2)
        """

@runtime_checkable
class UnbinnedThreeMLSourceResponseInterface(ThreeMLSourceResponseInterface, ExpectationDensityInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class CachedUnbinnedThreeMLSourceResponseInterface(UnbinnedThreeMLSourceResponseInterface, Protocol):
    """
    Guaranteeing that the source response can be cached to and loaded from a file.
    """
    
    def cache_to_file(self, filename: Union[str, Path]):
        """
        Saves the calculated response cache to the specified HDF5 file.
        The implementation has to make sure that the source is handled correctly.
        """

    def cache_from_file(self, filename: Union[str, Path]):
        """Loads the response cache from the specified HDF5 file."""
        
    def init_cache(self):
        """
        Initialize the response cache that can be saved to file.
        This way there is no need to call expected_counts() or expectation_density() to initialize the cache.
        Make sure that repeated calls don't lead to unnecessary recomputations.
        """

@runtime_checkable
class BinnedThreeMLSourceResponseInterface(ThreeMLSourceResponseInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """


