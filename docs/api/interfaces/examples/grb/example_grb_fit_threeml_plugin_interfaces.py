import logging

from cosipy.util import fetch_wasabi_file

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

import sys

from mhealpy import HealpixBase

from matplotlib import pyplot as plt

from cosipy.statistics import PoissonLikelihood

from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse

from astropy.time import Time
import astropy.units as u

import numpy as np

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter

from pathlib import Path

import os

def main():

    # Download data
    data_path = Path("")  # /path/to/files. Current dir by default
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_bkg_binned_data.hdf5', output=str(data_path / 'grb_bkg_binned_data.hdf5'), checksum = 'fce391a4b45624b25552c7d111945f60')
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5', output=str(data_path / 'grb_binned_data.hdf5'), checksum = 'fcf7022369b6fb378d67b780fc4b5db8')
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/bkg_binned_data_1s_local.hdf5', output=str(data_path / 'bkg_binned_data_1s_local.hdf5'), checksum = 'b842a7444e6fc1a5dd567b395c36ae7f')
    # fetch_wasabi_file('COSI-SMEX/develop/Data/Orientation/20280301_3_month_with_orbital_info.fits', output=str(data_path / '20280301_3_month_with_orbital_info.fits'), checksum = '5e69bc1d55fab9390f90635690f62896')
    # fetch_wasabi_file('COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip', output=str(data_path / 'SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip'), unzip = True, checksum = 'e8ff763c5d9e63d3797567a4a51d9eda')

    dr_path = data_path / "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5"  # path to detector response
    fetch_wasabi_file(
        'COSI-SMEX/develop/Data/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5',
        output=str(dr_path),
        checksum='eb72400a1279325e9404110f909c7785')

    # Set model to fit
    l = 93.
    b = -53.

    alpha = -1
    beta = -3
    xp = 450. * u.keV
    piv = 500. * u.keV
    K = 1 / u.cm / u.cm / u.s / u.keV

    spectrum = Band()
    spectrum.beta.min_value = -15.0
    spectrum.alpha.value = alpha
    spectrum.beta.value = beta
    spectrum.xp.value = xp.value
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.xp.unit = xp.unit
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    source = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                         l=l,  # Longitude (deg)
                         b=b,  # Latitude (deg)
                         spectral_shape=spectrum)  # Spectral model

    # Date preparation
    binned_data = BinnedData(data_path / "grb.yaml")
    binned_data.load_binned_data_from_hdf5(binned_data=data_path / "grb_bkg_binned_data.hdf5")

    bkg = BinnedData(data_path / "background.yaml")

    bkg.load_binned_data_from_hdf5(binned_data=data_path / "bkg_binned_data_1s_local.hdf5")

    bkg_tmin = 1842597310.0
    bkg_tmax = 1842597550.0
    bkg_min = np.where(bkg.binned_data.axes['Time'].edges.value == bkg_tmin)[0][0]
    bkg_max = np.where(bkg.binned_data.axes['Time'].edges.value == bkg_tmax)[0][0]
    bkg_dist = bkg.binned_data.slice[{'Time': slice(bkg_min, bkg_max)}].project('Em', 'Phi', 'PsiChi')

    tmin = Time(1842597410.0, format='unix')
    tmax = Time(1842597450.0, format='unix')
    ori = SpacecraftHistory.open(data_path / "20280301_3_month_with_orbital_info.fits", tmin, tmax)
    ori = ori.select_interval(tmin, tmax) # Function changed name during refactoring

    # Prepare instrument response
    dr = FullDetectorResponse.open(dr_path)

    # Workaround to avoid inf values. Out bkg should be smooth, but currently it's not.
    # Reproduces results before refactoring. It's not _exactly_ the same, since this fudge value was 1e-12, and
    # it was added to the expectation, not the normalized bkg
    bkg_dist += sys.float_info.min

    # ============ Interfaces ==============
    data = binned_data.get_em_cds()

    bkg = FreeNormBinnedBackground(bkg_dist,
                                   sc_history=ori,
                                   copy = False)

    instrument_response = BinnedInstrumentResponse(dr, data)

    # Currently using the same NnuLambda, Ei and Pol axes as the underlying FullDetectorResponse,
    # matching the behavior of v0.3. This is all the current BinnedInstrumentResponse can do.
    # In principle, this can be decoupled, and a BinnedInstrumentResponseInterface implementation
    # can provide the response for an arbitrary directions, Ei and Pol values.
    # NOTE: this is currently only implemented for data in local coords
    psr = BinnedThreeMLPointSourceResponse(data = data,
                                           instrument_response = instrument_response,
                                           sc_history=ori,
                                           energy_axis = dr.axes['Ei'],
                                           polarization_axis = dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                           nside = 2*data.axes['PsiChi'].nside)

    response = BinnedThreeMLModelFolding(data = data, point_source_response = psr)

    like_fun = PoissonLikelihood(data, response, bkg)

    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)

    # Nuisance parameter guess, bounds, etc.
    cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",  # background parameter
                                      1,
                                      unit  = u.Hz,# initial value of parameter
                                      min_value=0,  # minimum value of parameter
                                      max_value=5,  # maximum value of parameter
                                      delta=1e-3,  # initial step used by fitting engine
                                      )

    # ======== Interfaces end ==========

    # 3Ml fit. Same as before
    plugins = DataList(cosi)
    model = Model(source)  # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)
    like = JointLikelihood(model, plugins)
    like.fit()
    results = like.results
    print(results.display())


if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof.prof")
    exit()

    main()
