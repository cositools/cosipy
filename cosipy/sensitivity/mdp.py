from cosipy import SourceInjector
from cosipy.statistics import PoissonLikelihood
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData
from cosipy.polarization import PolarizationAxis
from cosipy.threeml.util import to_linear_polarization

from threeML import LinearPolarization, StokesPolarization, PointSource, Model, JointLikelihood, DataList
import numpy as np
from histpy import Histogram
import copy

import logging
logger = logging.getLogger(__name__)

def compute_mdp(n, model, background, bkg_parameter, sc_orientation, response_file, 
				response_pa_convention, response_frame='spacecraftframe', confidence=99):
	"""
	Calculates the minimum detectable polarization using the maximum 
	likelihood method.

	Parameters:
	n : int
		Number of simulations.
	model : :py:class:`astromodels.core.model.Model`
		Source model.
	background : :py:class:`cosipy.background_estimation`
		Background model.
	bkg_parameter : :py:class:`astromodels.core.parameter.Parameter`
		Background parameter.
	sc_orientation : :py:class:`cosipy.spacecraftfile.spacecraft_file.SpacecraftHistory`
		Spacecraft history.
	response_file : :py:class:`pathlib.PosixPath`
		Path to response file.
	response_pa_convention : str
		Polarization angle convention of response.
	response_frame : str, optional
		Frame of response file.
	confidence : int, optional
		Confidence level.
	"""

	if len(model.source.spectrum.to_dict()) > 1:
		raise RuntimeError('Model cannot contain more than one source.')

	source_direction = model.source.position.sky_coord

	for key in model.source.spectrum.to_dict().keys():
		spectrum = model.source._components[key].shape

	degrees = []
	angles = []
	failed_fits = 0

	dr = FullDetectorResponse.open(response_file, pa_convention=response_pa_convention)
	
	for i in range(n):

		this_model = copy.deepcopy(model)

		injector = SourceInjector(response_path=response_file,
								  response_frame=response_frame,
								  pa_convention=response_pa_convention)

		unpolarized = LinearPolarization(angle=0, degree=0)

		source = PointSource('source_mdp', 
							 l=source_direction.transform_to('galactic').l.deg, 
							 b=source_direction.transform_to('galactic').b.deg, 
							 spectral_shape=spectrum)

		model_inj = Model(source)

		data = injector.inject_model(model=model_inj,
									 orientation=sc_orientation,
									 polarization=unpolarized,
									 make_spectrum_plot=False,
									 earth_occ=True)

		data[:] = np.random.poisson(data, data.shape)
		bkg = Histogram(background.expectation().axes, contents=np.random.poisson(background.expectation(), background.expectation().shape))

		data += bkg
		data = EmCDSBinnedData(data)

		instrument_response = BinnedInstrumentResponse(dr, data)

		psr = BinnedThreeMLPointSourceResponse(data=data,
											   instrument_response=instrument_response,
											   sc_history=sc_orientation,
											   energy_axis=dr.axes['Ei'],
											   polarization_axis=PolarizationAxis(dr.axes['Pol'], convention=response_pa_convention),
											   nside=2*data.axes['PsiChi'].nside)

		response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)

		like_fun = PoissonLikelihood(data, response, background)

		cosi = ThreeMLPluginInterface('cosi_mdp',
									  like_fun,
									  response,
									  background)

		cosi.bkg_parameter[bkg_parameter.name] = bkg_parameter

		plugins = DataList(cosi)

		like = JointLikelihood(this_model, plugins, verbose=False)

		try:

			_ = like.fit(quiet=True)

		except:

			failed_fits += 1
			continue

		results = like.results

		for key in model.source.spectrum.to_dict().keys():

			linear_polarization = to_linear_polarization(results.optimized_model.source.spectrum[key].polarization)
			degree = linear_polarization.degree.value
			angle = linear_polarization.angle.value

		degrees.append(degree)
		angles.append(angle)

	mdp = np.percentile(degrees, confidence)
	logger.warning(f'{failed_fits}/{n} fits failed')

	return mdp, degrees, angles