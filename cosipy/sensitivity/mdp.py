from cosipy import SourceInjector
from cosipy.statistics import PoissonLikelihood
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData
from cosipy.polarization import PolarizationAxis

from threeML import LinearPolarization, PointSource, Model, JointLikelihood, DataList
import numpy as np
from histpy import Histogram

import logging
logger = logging.getLogger(__name__)

def compute_mdp(n, source_direction, spectrum, model, background, bkg_parameter, 
				fix_bkg, sc_orientation, response_file, response_pa_convention, 
				response_frame='spacecraftframe', confidence=99):
	"""
	Calculates the minimum detectable polarization using the maximum 
	likelihood method.

	Parameters:
	n : int
		Number of simulations.
	source_direction : :py:class:`astropy.coordinates.SkyCoord`
		Source direction.
	spectrum : :py:class:`threeML.Model`
		Source spectrum.
	model : :py:class:`astromodels.core.model.Model`
		Source model.
	background : :py:class:`cosipy.background_estimation`
		Background model.
	bkg_parameter : :py:class:`astromodels.core.parameter.Parameter`
		Background parameter.
	fix_bkg : bool
		Whether to fix background parameter.
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

	degrees = []
	failed_fits = 0

	dr = FullDetectorResponse.open(response_file, pa_convention=response_pa_convention)
	
	for i in range(n):

		injector = SourceInjector(response_path=response_file,
								  response_frame=response_frame,
								  pa_convention=response_pa_convention)

		unpolarized = LinearPolarization(angle=0, degree=0)

		source_mdp = PointSource('source_mdp', 
								 l=source_direction.transform_to('galactic').l.deg, 
								 b=source_direction.transform_to('galactic').b.deg, 
								 spectral_shape=spectrum)

		mdp_model = Model(source_mdp)

		mdp_data = injector.inject_model(model=mdp_model,
										 orientation=sc_orientation,
										 polarization=unpolarized,
										 make_spectrum_plot=False,
										 earth_occ=True)

		mdp_data[:] = np.random.poisson(mdp_data, mdp_data.shape)
		bkg = Histogram(background.expectation().axes, contents=np.random.poisson(background.expectation(), background.expectation().shape))

		mdp_data += bkg
		mdp_data = EmCDSBinnedData(mdp_data)

		mdp_instrument_response = BinnedInstrumentResponse(dr, mdp_data)

		mdp_psr = BinnedThreeMLPointSourceResponse(data=mdp_data,
												   instrument_response=mdp_instrument_response,
												   sc_history=sc_orientation,
												   energy_axis=dr.axes['Ei'],
												   polarization_axis=PolarizationAxis(dr.axes['Pol'], convention=response_pa_convention),
												   nside=2*mdp_data.axes['PsiChi'].nside)

		mdp_response = BinnedThreeMLModelFolding(data=mdp_data, point_source_response=mdp_psr)

		like_fun = PoissonLikelihood(mdp_data, mdp_response, background)

		mdp_cosi = ThreeMLPluginInterface('cosi_mdp',
										  like_fun,
										  mdp_response,
										  background)

		mdp_cosi.bkg_parameter[bkg_parameter.name] = bkg_parameter
		mdp_cosi.bkg_parameter[bkg_parameter.name].fix = fix_bkg

		mdp_plugins = DataList(mdp_cosi)

		mdp_like = JointLikelihood(model, mdp_plugins, verbose=False)

		try:

			_ = mdp_like.fit(quiet=True)

		except:

			failed_fits += 1
			continue

		results = mdp_like.results

		for key in model.source.spectrum.to_dict().keys():
			degree = results.optimized_model.source.spectrum[key].polarization.degree.value
		
		degrees.append(degree)

	mdp = np.percentile(degrees, confidence)
	logger.warning(f'{failed_fits}/{n} fits failed')

	return mdp