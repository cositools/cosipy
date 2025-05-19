from cosipy import COSILike, test_data, BinnedData
from cosipy.spacecraftfile import SpacecraftFile
import astropy.units as u
import numpy as np
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy.coordinates import SkyCoord

data_path = test_data.path

sc_orientation = SpacecraftFile.parse_from_file(data_path / "20280301_2s.ori")
dr = str(data_path / "test_full_detector_response.h5") # path to detector response

data = BinnedData(data_path / "test_spectral_fit.yaml")
background = BinnedData(data_path / "test_spectral_fit.yaml")

data.load_binned_data_from_hdf5(binned_data=data_path / "test_spectral_fit_data.h5")
background.load_binned_data_from_hdf5(binned_data=data_path / "test_spectral_fit_background.h5")

bkg_par = Parameter("background_cosi",                                         # background parameter
                    1,                                                         # initial value of parameter
                    min_value=0,                                               # minimum value of parameter
                    max_value=50,                                              # maximum value of parameter
                    delta=0.05,                                                # initial step used by fitting engine
                    desc="Background parameter for cosi")

l = 50
b = -45

alpha = -1
beta = -2
xp = 500. * u.keV
piv = 500. * u.keV
K = 1 / u.cm / u.cm / u.s / u.keV

spectrum = Band()

spectrum.alpha.value = alpha
spectrum.beta.value = beta
spectrum.xp.value = xp.value
spectrum.K.value = K.value
spectrum.piv.value = piv.value

spectrum.xp.unit = xp.unit
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

source = PointSource("source",                     # Name of source (arbitrary, but needs to be unique)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

model = Model(source)

def test_point_source_spectral_fit():

    cosi = COSILike("cosi",                                                        # COSI 3ML plugin
                    dr = dr,                                                       # detector response
                    data = data.binned_data.project('Em', 'Phi', 'PsiChi'),        # data (source+background)
                    bkg = background.binned_data.project('Em', 'Phi', 'PsiChi'),   # background model
                    sc_orientation = sc_orientation,                               # spacecraft orientation
                    nuisance_param = bkg_par)                                      # background parameter

    plugins = DataList(cosi)

    like = JointLikelihood(model, plugins, verbose = False)

    like.fit()

    assert np.allclose([source.spectrum.main.Band.K.value, source.spectrum.main.Band.alpha.value, source.spectrum.main.Band.beta.value, source.spectrum.main.Band.xp.value, bkg_par.value],
                       [1.0522695866399103,  2.6276132958523926, -2.909795888815157, 18.19702619330248, 2.3908438191547012], atol=[0.1, 0.1, 0.1, 1.0, 0.1])

    assert np.allclose([cosi.get_log_like()], [213.14242014103897], atol=[1.0])

    # Test scatt map method:
    coord = SkyCoord(l=184.56*u.deg,b=-5.78*u.deg,frame="galactic")
    cosi._get_scatt_map(coord)
