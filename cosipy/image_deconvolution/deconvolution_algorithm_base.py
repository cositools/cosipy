import gc
from tqdm.autonotebook import tqdm
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import angular_separation

from histpy import Histogram, Axes, Axis

class DeconvolutionAlgorithmBase(object):
    """
    A base class for image deconvolution algorithms.
    A subclass should override these methods:

    - pre_processing
    - Estep
    - Mstep
    - post_processing
    - check_stopping_criteria
    - register_result
    - save_result
    - show_result
    
    When the method run_deconvolution is called in ImageDeconvolution class, 
    the iteration method in this class is called for each iteration.

    Attributes
    ----------
    initial_model_map : :py:class:`cosipy.image_deconvolution.ModelMap`
        Initial values for reconstructed images
    data : :py:class:`cosipy.image_deconvolution.DataLoader`
        COSI data set 
    parameter : py:class:`cosipy.config.Configurator`
        Parameters for a deconvolution algorithm
    """

    def __init__(self, initial_model_map, data, parameter):

        self.data = data 

        self.parameter = parameter 

        self.initial_model_map = initial_model_map

        # image axis (model space)
        image_axis = initial_model_map.axes['lb']

        self.nside_model = image_axis.nside
        self.npix_model = image_axis.npix
        self.pixelarea_model = hp.nside2pixarea(self.nside_model) * u.sr

        # energy axis (model space)
        energy_axis = initial_model_map.axes['Ei']
        self.n_energyband_model = energy_axis.nbins

        # NuLambda axis (Compton data space)
        response_axis = data.full_detector_response.axes['NuLambda']

        self.nside_local = response_axis.nside
        self.npix_local = response_axis.npix
        self.pixelarea_local = hp.nside2pixarea(self.nside_local) * u.sr

        # reconstructed image and related data
        self.model_map = None
        self.delta_map = None
        self.processed_delta_map = None
        self.bkg_norm = 1.0

        self.result = None

        self.expectation = None
        
        # parameters of the iteration
        self.iteration_max = parameter['iteration']

        self.save_result = parameter.get("save_results_each_iteration", False)

    def pre_processing(self):
        """
        pre-processing for each iteration
        """
        pass

    def Estep(self):
        """
        E-step. Basically expected counts are calculated here (or at the end of iteration in some cases).
        """
        pass

    def Mstep(self):
        """
        M-step. Basically a first feedback to a model map (delta map) is calculated here.
        """
        pass

    def post_processing(self):
        """
        Post-processing. For example, filters like gaussian smoothing are applied to the delta map in this step.
        """
        pass

    def check_stopping_criteria(self, i_iteration):
        """
        Check whether iteration process should be continued or stopped.
        """
        if i_iteration < 0:
            return False
        return True

    def register_result(self, i_iteration):
        """
        Register results at the end of each iteration. Users can define what kinds of values will be stored in this method.
        """
        this_result = {"iteration": self.i_iteration + 1}
        self.result = this_result

    def save_result(self, i_iteration):
        """
        Save some results at the end of each iteration.
        """
        pass

    def show_result(self, i_iteration):
        """
        Show some results at the end of each iteration. 
        """
        pass

    def iteration(self):
        """
        Perform one iteration of image deconvolution.
        This method should not be overrided in subclasses.
        """
        self.model_map = self.initial_model_map

        stop_iteration = False
        for i_iteration in tqdm(range(1, self.iteration_max + 1)):
            gc.collect()

            if stop_iteration:
                break

            print("  Iteration {}/{} ".format(i_iteration, self.iteration_max))

            print("--> pre-processing")
            self.pre_processing()

            print("--> E-step")
            self.Estep()
            gc.collect()

            print("--> M-step")
            self.Mstep()
            gc.collect()
            
            print("--> post-processing")
            self.post_processing()

            print("--> checking stopping criteria")
            stop_iteration = self.check_stopping_criteria(i_iteration)
            print("--> --> {}".format("stop" if stop_iteration else "continue"))

            print("--> registering results")
            self.register_result(i_iteration)

            print("--> showing results")
            self.show_result(i_iteration)
            
            if self.save_result == True:
                print("--> saving results")
                self.save_result(i_iteration)

            gc.collect()

            yield self.result
    
    def calc_expectation(self, model_map, data, almost_zero = 1e-12):
        """
        Calculate expected counts from a given model map.

        Parameters
        ----------
        model_map : :py:class:`cosipy.image_deconvolution.ModelMap`
            Model map
        data : :py:class:`cosipy.image_deconvolution.DataLoader`
            COSI data set 
        almost_zero : float, default 1e-12
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Expected count histogram

        Notes
        -----
        This method should be implemented in a more general class, for example, extended source response class in the future.
        """
        # Currenly (2024-01-12) this method can work for both local coordinate CDS and in galactic coordinate CDS.
        # This is just because in DC2 the rotate response for galactic coordinate CDS does not have an axis for time/scatt binning.
        # However it is likely that it will have such an axis in the future in order to consider background variability depending on time and pointign direction etc.
        # Then, the implementation here will not work. Thus, keep in mind that we need to modify it once the response format is fixed.

        expectation = Histogram(data.event_dense.axes) 

        map_rotated = np.tensordot(data.coordsys_conv_matrix.contents, model_map.contents, axes = ([1], [0])) 
        # ['Time/ScAtt', 'lb', 'NuLambda'] x ['lb', 'Ei'] -> [Time/ScAtt, NuLambda, Ei]
        map_rotated *= data.coordsys_conv_matrix.unit * model_map.unit
        map_rotated *= self.pixelarea_model
        # data.coordsys_conv_matrix.contents is sparse, so the unit should be restored.
        # the unit of map_rotated is 1/cm2 ( = s * 1/cm2/s/sr * sr)

        if data.response_on_memory == True:
            expectation[:] = np.tensordot( map_rotated, data.image_response_dense.contents, axes = ([1,2], [0,1]))
        else:
            for ipix in tqdm(range(self.npix_local)):
                response_this_pix = np.sum(data.full_detector_response[ipix].to_dense(), axis = (4,5)) # ['Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'] -> ['Ei', 'Em', 'Phi', 'PsiChi']
                expectation += np.tensordot(map_rotated[:,ipix,:], response_this_pix, axes = ([1], [0]))

        expectation += data.bkg_dense * self.bkg_norm
        expectation += almost_zero
        
        return expectation

    def calc_loglikelihood(self, data, model_map, expectation = None):
        """
        Calculate log-likelihood from given data and model/expectation.

        Parameters
        ----------
        data : :py:class:`cosipy.image_deconvolution.DataLoader`
            COSI data set 
        model_map : :py:class:`cosipy.image_deconvolution.ModelMap`
            Model map
        expectation : :py:class:`histpy.Histogram` or None, default None
            Expected count histogram.
            If it is not given, expected counts will be calculated in this method.

        Returns
        -------
        float
            Log-likelood

        Notes
        -----
        The parameter expectation may be a mandatory parameter in the future.
        """
        if expectation is None:
            expectation = self.calc_expectation(model_map, data)

        loglikelood = np.sum( data.event_dense * np.log(expectation) ) - np.sum(expectation)

        return loglikelood

    def calc_gaussian_filter(self, sigma):
        """
        Calculate a gaussian filter for post-processing.

        Parameters
        ----------
        sigma: float
            Sigma for Gaussian function. It should be in degrees, but unitless.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Gaussian filter. 2-dimensional histogram.
        """

        gaussian_filter = Histogram( Axes( [Axis(edges = np.arange(self.npix_model+1)), Axis(edges = np.arange(self.npix_model+1))] ), sparse = False)

        for ipix in tqdm(range(self.npix_model)):

            lon_ref, lat_ref = hp.pix2ang(self.nside_model, ipix, nest = False, lonlat = True)

            lon, lat = hp.pix2ang(self.nside_model, np.arange(self.npix_model), nest = False, lonlat = True)

            delta_ang = angular_separation(lon_ref * u.deg, lat_ref * u.deg, lon * u.deg, lat * u.deg).to('deg').value
    
            gaussian_filter[ipix,:] = np.exp( - 0.5 * delta_ang**2 / sigma**2)
    
            gaussian_filter[ipix,:] /= np.sum(gaussian_filter[ipix,:])  

        return gaussian_filter
