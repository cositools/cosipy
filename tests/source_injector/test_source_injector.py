from cosipy import SpacecraftHistory, SourceInjector
from astropy.coordinates import SkyCoord
from threeML import Powerlaw
from pathlib import Path
import os
from cosipy import test_data
import numpy as np
import astropy.units as u
from histpy import Histogram
import pytest
from astromodels import Model, PointSource, ExtendedSource, Powerlaw, Gaussian_on_sphere

def test_inject_point_source():

    # defind the response and orientation
    response_path = test_data.path / "test_full_detector_response.h5"
    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftHistory.open(orientation_path)

    # powerlaw model
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    # Define an injector by the response
    injector = SourceInjector(response_path = response_path)

    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    # Get the data of the injected source
    injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord,
                                                        orientation = ori, source_name = "point_source",
                                                        make_spectrum_plot = False, make_PsiChi_plot = False ,data_save_path = None,
                                                        project_axes = None)

    results = injected_crab_signal.project("Em").to_dense().contents

    assert isinstance(results, u.quantity.Quantity) == True

    assert np.allclose(results.value,
                       [5.18769386e-01, 1.07545259e+00, 8.66760819e-01, 4.54548331e-01,
                        2.18439534e-01, 1.03093234e-01, 4.93963707e-02, 1.64003979e-02,
                        3.07634751e-03, 1.44128663e-04])


def test_inject_point_source_galactic():

    # defind the response and orientation
    response_path = test_data.path / "test_precomputed_response.h5"

    # powerlaw model
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    # Define an injector by the response
    injector = SourceInjector(response_path = response_path, response_frame = "galactic")

    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    # Get the data of the injected source
    injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord,
                                                        source_name = "point_source",
                                                        make_spectrum_plot = True, make_PsiChi_plot=True , data_save_path = None,
                                                        project_axes = None)

    results = injected_crab_signal.project("Em").to_dense().contents

    assert isinstance(results, u.quantity.Quantity) == True

    assert np.allclose(results.value,
                       [8.00446239e-02, 2.39541274e-01, 3.06395646e-01, 2.90215536e-01,
                        2.18503792e-01, 1.41017794e-01, 8.27902948e-02, 3.04628607e-02,
                        5.75082017e-03, 2.48831060e-04])


def test_inject_point_source_saving():

    # defind the response and orientation
    response_path = test_data.path / "test_precomputed_response.h5"

    # powerlaw model
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    # Define an injector by the response
    injector = SourceInjector(response_path = response_path, response_frame = "galactic")

    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    # Get the data of the injected source
    injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord,
                                                        source_name = "point_source",
                                                        make_spectrum_plot = False, make_PsiChi_plot=False ,data_save_path = Path("./galactic_rsp.h5"),
                                                        project_axes = "Em")

    hist= Histogram.open(Path("./galactic_rsp.h5"))

    os.remove(Path("./galactic_rsp.h5"))

    assert np.allclose(hist[:].value,
                       [8.00446239e-02, 2.39541274e-01, 3.06395646e-01, 2.90215536e-01,
                        2.18503792e-01, 1.41017794e-01, 8.27902948e-02, 3.04628607e-02,
                        5.75082017e-03, 2.48831060e-04])


def test_response_frame_error():

    # defind the response and orientation
    response_path = test_data.path / "test_precomputed_response.h5"

    with pytest.raises(ValueError):
        injector = SourceInjector(response_path = response_path, response_frame = "some_frame")


def test_orientation_error():

    # defind the response and orientation
    response_path = test_data.path / "test_full_detector_response.h5"

    # powerlaw model
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    # Define an injector by the response
    injector = SourceInjector(response_path = response_path)

    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    with pytest.raises(TypeError):

        # Get the data of the injected source
        injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord,
                                                            source_name = "point_source",
                                                            make_spectrum_plot = False, make_PsiChi_plot=False ,data_save_path = None,
                                                            project_axes = None)


def test_inject_extended_source():

    # Define the response
    response_path = test_data.path / "test_precomputed_response.h5"

    # Define a spatial model (Gaussian_on_sphere) + spectral model (Powerlaw)
    spatial = Gaussian_on_sphere()
    spatial.lon0.value = 50.0
    spatial.lat0.value = -45.0
    spatial.sigma.value = 2.0

    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV

    spectral = Powerlaw()
    spectral.index.value = -2.2
    spectral.K.value = K.value
    spectral.piv.value = piv.value
    spectral.K.unit = K.unit
    spectral.piv.unit = piv.unit

    # Combine into an ExtendedSource model
    model = ExtendedSource(
        "test_extended", spatial_shape=spatial, spectral_shape=spectral
    )

    # Define an injector by the response
    injector = SourceInjector(response_path=response_path)

    # Get the data of the injected source
    injected = injector.inject_extended_source(
        source_model=model,
        make_spectrum_plot=True,
        make_PsiChi_plot=True,
        data_save_path=None,
        project_axes=None,
    )

    hist = injected.project("Em").to_dense().contents

    assert isinstance(hist, u.quantity.Quantity) == True
    assert np.sum(hist.value) > 0  # ensure there is some non-zero expectation


def test_inject_extended_source_saving():

    # Define the response
    response_path = test_data.path / "test_precomputed_response.h5"

    # Define a spatial model (Gaussian_on_sphere) + spectral model (Powerlaw)
    spatial = Gaussian_on_sphere()
    spatial.lon0.value = 50.0
    spatial.lat0.value = -45.0
    spatial.sigma.value = 2.0

    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV

    spectral = Powerlaw()
    spectral.index.value = -2.2
    spectral.K.value = K.value
    spectral.piv.value = piv.value
    spectral.K.unit = K.unit
    spectral.piv.unit = piv.unit

    model = ExtendedSource(
        "test_extended", spatial_shape=spatial, spectral_shape=spectral
    )

    # Define an injector by the response
    injector = SourceInjector(response_path=response_path)

    file_path = Path("./extended_rsp.h5")

    # Get the data of the injected source
    injected = injector.inject_extended_source(
        source_model=model,
        make_spectrum_plot=False,
        make_PsiChi_plot=False,
        data_save_path=file_path,
        project_axes=None,
    )

    hist = Histogram.open(file_path)
    os.remove(file_path)

    assert np.sum(hist[:].value) > 0  # ensure there is some non-zero expectation


def test_get_esr_error():

    # Define an invalid response
    response_path = test_data.path / "invalid_response.h5"

    # Define a spatial model (Gaussian_on_sphere) + spectral model (Powerlaw)
    spatial = Gaussian_on_sphere()
    spatial.lon0.value = 50.0
    spatial.lat0.value = -45.0
    spatial.sigma.value = 2.0

    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV

    spectral = Powerlaw()
    spectral.index.value = -2.2
    spectral.K.value = K.value
    spectral.piv.value = piv.value
    spectral.K.unit = K.unit
    spectral.piv.unit = piv.unit

    # Get the data of the injected source
    model = ExtendedSource(
        "test_extended", spatial_shape=spatial, spectral_shape=spectral
    )

    with pytest.raises(RuntimeError): # Expect RuntimeError for invalid response file
        SourceInjector.get_esr(model, response_path)


def test_inject_model():

    # Define the response
    response_path = test_data.path / "test_precomputed_response.h5"
    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftHistory.open(orientation_path)

    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV

    spectral = Powerlaw()
    spectral.index.value = -2.2
    spectral.K.value = K.value
    spectral.piv.value = piv.value
    spectral.K.unit = K.unit
    spectral.piv.unit = piv.unit

    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    c_icrs = source_coord.transform_to('icrs')
    model_point = PointSource("test_point",
                              ra = c_icrs.ra.deg,
                              dec = c_icrs.dec.deg,
                              spectral_shape = spectral)

    # Define a spatial model (Gaussian_on_sphere) + spectral model (Powerlaw)
    spatial = Gaussian_on_sphere()
    spatial.lon0.value = 50.0
    spatial.lat0.value = -45.0
    spatial.sigma.value = 2.0

    model_ext = ExtendedSource(
        "test_extended",
        spatial_shape=spatial,
        spectral_shape=spectral
    )

    model = Model(model_point, model_ext)

    # Define an injector by the response
    injector = SourceInjector(response_path=response_path,
                              response_frame="galactic")

    file_path = Path("./combined_rsp.h5")

    # Get the data of the injected source
    injected = injector.inject_model(model,
                                     data_save_path=file_path)

    hist = Histogram.open(file_path)
    os.remove(file_path)

    hist = injected.project("Em").contents

    assert isinstance(hist, u.quantity.Quantity)
    assert np.sum(hist.value) > 0  # ensure there is some non-zero expectation
