#!/usr/bin/env python
# coding: utf-8

import logging

from astropy.utils.metadata.utils import dtype
from histpy import Histogram, HealpixAxis
from mhealpy import HealpixMap

from cosipy.background_estimation.free_norm_threeml_binned_bkg import FreeNormBackgroundInterpolatedDensityTimeTagEmCDS
from cosipy.event_selection import GoodTimeInterval
from cosipy.interfaces.expectation_interface import SumExpectationDensity
from cosipy.threeml.unbinned_model_folding import UnbinnedThreeMLModelFolding

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import cProfile

from cosipy import test_data, BinnedData, UnBinnedData
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromDC3Fits
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface
from cosipy.response.instrument_response_function import UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.threeml.psr_fixed_ei import UnbinnedThreeMLPointSourceResponseTrapz
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood, UnbinnedLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse

import sys

from scoords import SpacecraftFrame

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, Angle, UnitSphericalRepresentation, CartesianRepresentation, \
    angular_separation

import numpy as np
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Powerlaw

from pathlib import Path

import os

def main():

    use_bkg = True

    profile = cProfile.Profile()

    # Download all data
    data_path = Path("")  # /path/to/files. Current dir by default

    crab_data_path = data_path / "crab_standard_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Sources/crab_standard_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
                      output=str(crab_data_path), checksum='1d73e7b9e46e51215738075e91a52632')

    bkg_data_path = data_path / "AlbedoPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Backgrounds/Ge/AlbedoPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
                      output=str(bkg_data_path), checksum='191a451ee597fd2e4b1cf237fc72e6e2')

    dr_path = data_path / "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5"  # path to detector response
    fetch_wasabi_file(
        'COSI-SMEX/develop/Data/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5',
        output=str(dr_path),
        checksum='eb72400a1279325e9404110f909c7785')

    sc_orientation_path = data_path / "DC3_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
    fetch_wasabi_file('COSI-SMEX/develop/Data/Orientation/DC3_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits',
                      output=str(sc_orientation_path), checksum='1b851c042acf4c909798e2401e9d2e38')

    binned_bkg_data_path = data_path / "bkg_binned_data.hdf5"
    fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/bkg_binned_data.hdf5',
                    output=str(binned_bkg_data_path), checksum = '54221d8556eb4ef520ef61da8083e7f4')

    # orientation history
    # About 1 full orbit ~1.7 hr
    tstart = Time("2028-03-01 02:00:00.117")
    tstop = Time("2028-03-01 03:42:00.117")
    gti = GoodTimeInterval(tstart, tstop)
    sc_orientation = SpacecraftHistory.open(sc_orientation_path)
    sc_orientation = sc_orientation.apply_gti(gti)

    # Prepare instrument response function
    logger.info("Loading response....")
    dr = FullDetectorResponse.open(dr_path)
    irf = UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction(dr)
    logger.info("Loading response DONE")

    # Prepare data
    selector = TimeSelector(tstart = sc_orientation.tstart, tstop = sc_orientation.tstop)

    logger.info("Loading data...")
    if use_bkg:
        data_file = [crab_data_path, bkg_data_path]
    else:
        data_file = crab_data_path

    data = TimeTagEmCDSEventDataInSCFrameFromDC3Fits(data_file,
                                                     selection=selector)

    logger.info("Loading data DONE")

    # Set background

    if use_bkg:
        bkg = BinnedData(data_path / "background.yaml")
        bkg.load_binned_data_from_hdf5(binned_data=str(binned_bkg_data_path))
        bkg_dist = bkg.binned_data.project('Em', 'Phi', 'PsiChi')

        # Workaround to avoid inf values. Our bkg should be smooth, but currently it's not.
        bkg_dist += sys.float_info.min

        logger.info("Setting bkg...")
        bkg = FreeNormBackgroundInterpolatedDensityTimeTagEmCDS(data, bkg_dist, sc_orientation, copy = False)
        bkg.set_norm(5*u.Hz)
        logger.info("Setting bkg DONE")
    else:
        bkg = None

    # Prepare point source response, which convolved the IRF with the SC orientation
    ei_samples = np.geomspace(100, 5000, 100)*u.keV
    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_orientation,
                                                  ei_samples)

    # Prepare the model
    l = 184.56
    b = -5.78

    index = -2.26
    piv = 1 * u.MeV
    K = 3e-6 / u.cm / u.cm / u.s / u.keV

    spectrum = Powerlaw()

    spectrum.index.min_value = -3
    spectrum.index.max_value = -1

    # Fix it for testing purposes
    spectrum.index.free = True

    spectrum.K.value = K.value
    spectrum.piv.value = piv.value

    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    spectrum.index.delta = 0.01

    source = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                         l=l,  # Longitude (deg)
                         b=b,  # Latitude (deg)
                         spectral_shape=spectrum)  # Spectral model

    model = Model(
        source)  # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)


    # Set model folding
    response = UnbinnedThreeMLModelFolding(psr)

    # response.set_model(model) # optional. Will be called by likelihood
    # print(response.ncounts())
    # print(np.fromiter(response.expectation_density(), dtype = float))

    # Setup likelihood
    if use_bkg:
        expectation_density = SumExpectationDensity(response, bkg)
    else:
        expectation_density = response

    # Test plots. REMOVE
    # response.set_model(model)
    # exdenlist = np.fromiter(expectation_density.expectation_density(), dtype=float)

    # plot expectation density energy
    # energy = np.fromiter([e.energy_keV for e in data], dtype = float)
    # fig,ax = plt.subplots()
    # ax.scatter(energy, exdenlist)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # h = Histogram(np.geomspace(50,5000))
    # h.fill(energy)
    # h /= h.axis.widths
    # h *= np.max(exdenlist) / np.max(h)
    # h.plot(ax)
    # plt.show()

    # plot expectation density phi
    # phi = np.fromiter([e.scattering_angle_rad for e in data], dtype = float)
    # phi *= 180/3.1416
    # fig,ax = plt.subplots()
    # ax.scatter(phi, exdenlist)
    # h = Histogram(np.linspace(0,180))
    # h.fill(phi)
    # h /= h.axis.widths
    # h *= np.max(exdenlist) / np.max(h)
    # h.plot(ax)
    # plt.show()

    # Plot ARM
    # attitudes = sc_orientation.interp_attitude(data.time)

    # psichi_sc = data.scattered_direction_sc.represent_as(UnitSphericalRepresentation)
    # coord_vec = source.position.sky_coord.transform_to(sc_orientation.attitude.frame).cartesian.xyz.value
    # sc_coord_vec = attitudes.rot.inv().apply(coord_vec)
    # sc_coord_sph = UnitSphericalRepresentation.from_cartesian(CartesianRepresentation(*sc_coord_vec.transpose()))
    # arm = angular_separation(sc_coord_sph.lon, sc_coord_sph.lat, psichi_sc.lon, psichi_sc.lat).to_value(u.deg) - phi
    #

    # psichi_sc = data.scattered_direction_sc.represent_as(UnitSphericalRepresentation)
    # psichi_sc_vec = psichi_sc.to_cartesian().xyz.value
    # psichi_gal_vec = attitudes.rot.apply(psichi_sc_vec.transpose())
    # psichi_coord = SkyCoord(CartesianRepresentation(*psichi_gal_vec.transpose()), frame = attitudes.frame)
    # arm = source.position.sky_coord.separation(psichi_coord).to_value(u.deg) - phi
    #
    # h = Histogram(np.linspace(-90,90,360))
    #
    # fig,ax = plt.subplots()
    # ax.scatter(arm, exdenlist)
    #
    # h.fill(arm)
    #
    # h_ex = Histogram(h.axis)
    # h_ex.fill(arm, weight=exdenlist)
    # h_ex /= h # Mean
    #
    # h /= h.axis.widths
    # h *= np.nanmax(h_ex) / np.max(h) # Normalize
    #
    # h.plot(ax, color = 'green')
    # h_ex.plot(ax, color='red')
    #
    # plt.show()

    # Plot CDS
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='mollview')
    #
    # sc = ax.scatter(psichi_coord.l.deg, psichi_coord.b.deg, transform=ax.get_transform('world'),
    #                 c = phi ,
    #                 cmap='inferno',
    #                 s=2, vmin=0, vmax=180)
    #
    # ax.scatter(source.position.sky_coord.l.deg, source.position.sky_coord.b.deg, transform=ax.get_transform('world'), marker='x', s=100, c='red')
    #
    # fig.colorbar(sc, fraction=.02, label="$\phi$ [deg]")
    #
    # m = HealpixMap(nside=128, coordsys='galactic')
    # m[:] = source.position.sky_coord.separation(m.pix2skycoord(np.arange(m.npix))).to_value(u.deg)
    # img = m.get_wcs_img(ax, coord='C') #Use C for a "bug" in healpy (doesn't work the same as plot()
    # ax.contour(img, levels=np.arange(0, 180, 10), cmap='inferno',
    #                 vmin=0, vmax=180)
    # plt.show()



    like_fun = UnbinnedLikelihood(expectation_density)

    cosi = ThreeMLPluginInterface('cosi', like_fun, response, bkg)

    # Nuisance parameter guess, bounds, etc.
    if use_bkg:
        cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",  # background parameter
                                          2.5,  # initial value of parameter
                                          unit = u.Hz,
                                          min_value=0,  # minimum value of parameter
                                          max_value=100,  # maximum value of parameter
                                          delta=0.05,  # initial step used by fitting engine
                                          )

    plugins = DataList(cosi) # If we had multiple instruments, we would do e.g. DataList(cosi, lat, hawc, ...)

    like = JointLikelihood(model, plugins, verbose = False)

    # Run
    print(data.nevents, expectation_density.expected_counts())
    profile.enable()
    like.fit()
    profile.disable()
    profile.dump_stats("prof_interfaces.prof")

    results = like.results

    # Plot the fitted and injected spectra

    # In[14]:


    fig, ax = plt.subplots()

    alpha_inj = -1.99
    beta_inj = -2.32
    E0_inj = 531. * (alpha_inj - beta_inj) * u.keV
    xp_inj = E0_inj * (alpha_inj + 2) / (alpha_inj - beta_inj)
    piv_inj = 100. * u.keV
    K_inj = 7.56e-4 / u.cm / u.cm / u.s / u.keV

    spectrum_inj = Band()

    spectrum_inj.alpha.min_value = -2.14
    spectrum_inj.alpha.max_value = 3.0
    spectrum_inj.beta.min_value = -5.0
    spectrum_inj.beta.max_value = -2.15
    spectrum_inj.xp.min_value = 1.0

    spectrum_inj.alpha.value = alpha_inj
    spectrum_inj.beta.value = beta_inj
    spectrum_inj.xp.value = xp_inj.value
    spectrum_inj.K.value = K_inj.value
    spectrum_inj.piv.value = piv_inj.value

    spectrum_inj.xp.unit = xp_inj.unit
    spectrum_inj.K.unit = K_inj.unit
    spectrum_inj.piv.unit = piv_inj.unit

    energy = np.geomspace(100 * u.keV, 10 * u.MeV).to_value(u.keV)

    flux_lo = np.zeros_like(energy)
    flux_median = np.zeros_like(energy)
    flux_hi = np.zeros_like(energy)
    flux_inj = np.zeros_like(energy)

    parameters = {par.name: results.get_variates(par.path)
                  for par in results.optimized_model["source"].parameters.values()
                  if par.free}

    results_err = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **parameters)

    for i, e in enumerate(energy):
        flux = results_err(e)
        flux_median[i] = flux.median
        flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
        flux_inj[i] = spectrum_inj.evaluate_at(e)

    ax.plot(energy, energy * energy * flux_median, label="Best fit")
    ax.fill_between(energy, energy * energy * flux_lo, energy * energy * flux_hi, alpha=.5, label="Best fit (errors)")
    ax.plot(energy, energy * energy * flux_inj, color='black', ls=":", label="Injected")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")

    ax.legend()

    ax.set_ylim(.1,100)

    plt.show()

    # Grid
    if use_bkg:
        loglike = Histogram([np.geomspace(5e-6, 15e-6, 30),
                                   np.geomspace(4, 5, 31)], labels=['K', 'B'], axis_scale='log')

        for i, k in enumerate(loglike.axes['K'].centers):
            for j, b in enumerate(loglike.axes['B'].centers):
                spectrum.K.value = k
                cosi.bkg_parameter['bkg_norm'].value = b

                loglike[i, j] = cosi.get_log_like()

    else:
        loglike = Histogram([np.geomspace(2e-6, 2e-4, 30)], labels=['K'], axis_scale='log')

        for i, k in enumerate(loglike.axes['K'].centers):
            spectrum.K.value = k

            loglike[i] = cosi.get_log_like()

    ax, plot = loglike.plot(vmin = np.max(loglike) - 25, vmax = np.max(loglike))

    plt.show()

    return


if __name__ == "__main__":

    main()
