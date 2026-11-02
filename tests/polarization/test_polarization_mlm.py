from astropy.coordinates import SkyCoord
from astropy import units as u
from threeML import LinearPolarization, StokesPolarization, SpectralComponent, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Constant
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
from cosipy.sensitivity.mdp import compute_mdp
from cosipy import test_data
from cosipy.threeml.util import to_linear_polarization

analysis = BinnedData(test_data.path / 'polarization_data_mlm.yaml')
analysis.load_binned_data_from_hdf5(test_data.path / 'polarization_data_binned.hdf5')
response_file = test_data.path / 'test_polarization_response.h5'
dr = FullDetectorResponse.open(response_file, pa_convention='RelativeZ')
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
Q = polarization.degree.value / 100. * np.cos(2. * polarization.angle.value * np.pi / 180.)
U = polarization.degree.value / 100. * np.sin(2. * polarization.angle.value * np.pi / 180.)
polarization = StokesPolarization(Q=Constant(k=Q), U=Constant(k=U))
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
                                       polarization_axis = dr.axes['Pol'],
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

	assert np.allclose([source.spectrum.test.polarization.Q.Constant.k.value, source.spectrum.test.polarization.U.Constant.k.value],
					   [.74, 0.], atol=[.2, .2])

def test_mdp():

	spectral_component_mdp = SpectralComponent('test_mdp', spectrum, polarization)

	source_mdp = PointSource('source',
							 l = source_direction.l.deg,        
							 b = source_direction.b.deg,
							 components = [spectral_component_mdp])

	source_mdp.components['test_mdp'].shape.K.fix = True
	source_mdp.components['test_mdp'].shape.E0.fix = True
	source_mdp.components['test_mdp'].shape.alpha.fix = True
	source_mdp.components['test_mdp'].shape.beta.fix = True

	model_mdp = Model(source_mdp)

	bkg_parameter = Parameter('total_bkg',
							  0.0016,
							  min_value=0,
							  max_value=100,
							  delta=0.05,
							  unit=u.Hz,
							  free=False)

	mdp = compute_mdp(50, model_mdp, bkg, bkg_parameter, sc_orientation, response_file, 'RelativeZ')

	assert np.allclose([mdp], [25.], atol=[15.])
