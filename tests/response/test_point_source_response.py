import numpy as np
from numpy import array_equal as arr_eq
from histpy import Histogram, Axes, Axis
from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py as h5
from astropy.time import Time
from mhealpy import HealpixBase, HealpixMap

from cosipy import test_data
from cosipy.response.FullDetectorResponse import cosi_response
from cosipy.response import PointSourceResponse, FullDetectorResponse

from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, Model 
from threeML import StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian, TbAbs
from cosipy.threeml.custom_functions import SpecFromDat

import pytest

# init/load
response_path = test_data.path/"test_full_detector_response.h5"

with FullDetectorResponse.open(response_path) as response:
    exposure_map = HealpixMap(base=response,
                                    unit=u.s,
                                    coordsys=SpacecraftFrame())

    ti = Time('1999-01-01T00:00:00.123456789')
    tf = Time('2010-01-01T00:00:00')
    dt = (tf-ti).to(u.s)

    exposure_map[:4] = dt/4

    psr = response.get_point_source_response(exposure_map = exposure_map)

def test_photon_energy_axis():
    assert psr.photon_energy_axis == psr.axes['Ei']

def test_get_expectation():
    # supported spectral functions
    supported_spectral_functions = [Constant(), Line(), Quadratic(), Cubic(), Quartic(), StepFunction(), \
                                    StepFunctionUpper(), Cosine_Prior(), Uniform_prior(), DiracDelta(), PhAbs(), Gaussian()]
    for func in supported_spectral_functions:
        expectation = psr.get_expectation(func)
    # example implicitly supported :py:class:`threeML.Model` with units
    expectation = psr.get_expectation(TbAbs())
    # generic unsupported :py:class:`threeML.Model` without units
    with pytest.raises(RuntimeError) as pytest_wrapped_exp:
        expectation = psr.get_expectation(Model())
    assert pytest_wrapped_exp.type == RuntimeError