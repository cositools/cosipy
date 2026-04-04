from astropy.coordinates import SkyCoord
from astropy import units as u
from threeML import LinearPolarization, SpectralComponent, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from scoords import SpacecraftFrame
import numpy as np
import sys

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy.polarization import PolarizationAxis
from cosipy import test_data

analysis = BinnedData(test_data.path / 'polarization_data_mlm.yaml')
analysis.load_binned_data_from_hdf5(test_data.path / 'polarization_data_binned.hdf5')
dr = FullDetectorResponse.open(test_data.path / 'test_polarization_response.h5', pa_convention='RelativeZ')
sc_orientation = SpacecraftHistory.open(test_data.path / 'polarization_ori.fits')
attitude = sc_orientation.attitude[0]

a = 10. * u.keV
b = 10000. * u.keV
alpha = -1.
beta = -2.
ebreak = 350. * u.keV
K = 50. / u.cm / u.cm / u.s
spectrum = Band_Eflux(a = a.value,
                      b = b.value,
                      alpha = alpha,
                      beta = beta,
                      E0 = ebreak.value,
                      K = K.value)
spectrum.a.unit = a.unit
spectrum.b.unit = b.unit
spectrum.E0.unit = ebreak.unit
spectrum.K.unit = K.unit

source_direction = SkyCoord(0, 70, representation_type='spherical', frame=SpacecraftFrame(attitude=attitude), unit=u.deg).transform_to('galactic')

polarization = LinearPolarization(0.5, 100)
spectral_component = SpectralComponent('test', spectrum, polarization)

source = PointSource('test',
                     l = source_direction.l.deg,
                     b = source_direction.b.deg,
                     components = [spectral_component])

source.components['test'].shape.K.fix = True
source.components['test'].shape.E0.fix = True
source.components['test'].shape.alpha.fix = True
source.components['test'].shape.beta.fix = True

model = Model(source)

data = EmCDSBinnedData(analysis.binned_data.project('Em', 'Phi', 'PsiChi'))

total_bkg = analysis.binned_data.project('Em', 'Phi', 'PsiChi') * 0
bkg_dist = {'total_bkg':total_bkg+sys.float_info.min}
bkg = FreeNormBinnedBackground(bkg_dist, sc_history = sc_orientation, copy = False)

instrument_response = BinnedInstrumentResponse(dr, data)

psr = BinnedThreeMLPointSourceResponse(data = data,
                                       instrument_response = instrument_response,
                                       sc_history = sc_orientation,
                                       energy_axis = dr.axes['Ei'],
                                       polarization_axis = PolarizationAxis(dr.axes['Pol'], convention='RelativeZ'),
                                       nside = 2*data.axes['PsiChi'].nside)

response = BinnedThreeMLModelFolding(data = data, point_source_response = psr)

like_fun = PoissonLikelihood(data, response, bkg)

def test_polarization_fit():

    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)

    cosi.bkg_parameter['total_bkg'] = Parameter('total_bkg',
                                                0.0016,  
                                                min_value=0,  
                                                max_value=100,  
                                                delta=0.05,  
                                                unit = u.Hz)

    cosi.bkg_parameter['total_bkg'].fix = True

    plugins = DataList(cosi)

    like = JointLikelihood(model, plugins, verbose=False)

    _ = like.fit()

    assert np.allclose([source.spectrum.test.polarization.degree.value, source.spectrum.test.polarization.angle.value],
                       [83.8, 115.9], atol=[1., 1.])
