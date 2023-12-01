from threeML import PluginPrototype
from threeML.minimizer import minimization
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import FitFailed
from astromodels import Parameter

from cosipy.response.FullDetectorResponse import FullDetectorResponse

from scoords import SpacecraftFrame, Attitude

from mhealpy import HealpixMap

from cosipy.response import PointSourceResponse
from histpy import Histogram
import h5py as h5
from histpy import Axis, Axes
import sys

import astropy.units as u

import numpy as np
import sparse

from scipy.special import factorial

import collections

import copy

import logging
logger = logging.getLogger(__name__)

class COSILike(PluginPrototype):
    def __init__(self, name, dr, data, bkg, sc_orientation, nuisance_param=None, **kwargs):
        """
        COSI 3ML plugin
        
        Parameters
        ----------
        name : str
            Plugin name e.g. "cosi". Needs to have a distinct name with respect to other plugins in the same analysis
        dr : Path
            Path to full detector response
        data: histpy.Histogram
            Binned data. Note: Eventually this should be a cosipy data class
        bkg: histpy.Histogram
            Binned background model. Note: Eventually this should be a cosipy data class
        sc_orientation: cosipy.spacecraftfile.SpacecraftFile
            It contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
            the spacecraft for the duration of the data included in the analysis.
            orientation module
        """
        
        # create the hash for the nuisance parameters. We have none for now.
        self._nuisance_parameters = collections.OrderedDict()

        # call the prototype constructor. Boilerplate.
        super(COSILike, self).__init__(name, self._nuisance_parameters)

        # User inputs needed to compute the likelihood
        self._name = name
        self._rsp_path = dr
        self._dr = FullDetectorResponse.open(dr)
        self._data = data
        self._bkg = bkg
        self._sc_orientation = sc_orientation
    
        # Place-holder for cached data.
        self._model = None
        self._source = None
        self._psr = None
        self._signal = None
        
        # Set to fit nuisance parameter if given by user
        if nuisance_param == None:
            self.set_inner_minimization(False)
        elif isinstance(nuisance_param, Parameter):
            self.set_inner_minimization(True)
            self._bkg_par = nuisance_param
            self._nuisance_parameters[self._bkg_par.name] = self._bkg_par
            self._nuisance_parameters[self._bkg_par.name].free = self._fit_nuisance_params
        else:
            raise RuntimeError("Nuisance parameter must be astromodels.core.parameter.Parameter object")
        
    def set_model(self, model):
        
        """
        Set the model to be used in the joint minimization.
        
        Parameters:
            model: LikelihoodModel
                Any model supported by astromodel. 
                Currently supports point sources or extended sources.
                 - Can't yet do both simultaneously.
                 - Only one point source allowed. 
                 - Can fit multiple extended sources at once.
        """
        
        # Get point sources and extended sources from model: 
        point_sources = model.point_sources
        extended_sources = model.extended_sources

        # Source counter for models with multiple sources: 
        # Note: Only for extended sources right now.
        self.src_counter = 0
        
        # Get expectation for extended sources:
        for name,source in extended_sources.items():
            
            # Set spectrum:
            # Note: the spectral parameters are updated internally by 3ML
            # during the likelihood scan. 
            spectrum = source.spectrum.main.shape
            
            # Testing only:
            #this_pix = self.grid[0]
            #weight = self.skymap1[this_pix]*4*np.pi/self.skymap1.npix
            #total_expectation = self.grid_response.get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi']) * weight

            # Get expectation (method 1 -- on the fly):
            #for i in range(0,len(self.grid_response)):

                # Get weight
            #    this_pix = self.grid[i]
            #    weight = self.skymap1[this_pix]*4*np.pi/self.skymap1.npix

            #    if i == 0:
            #        total_expectation = self.grid_response[i].get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi']) * weight
            #    if i > 0:
            #        this_expectation = self.grid_response[i].get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi']) * weight
            #        total_expectation += this_expectation
            
            # Get expectation (method 2 -- precomputed psr in Galactic coordinates):
            total_expectation = Histogram(edges = self.psr_axes[2:])

            with h5.File(self.response_file) as f:

                #for pix,weight in enumerate(self.skymap1): # using full sky
                for i in range(len(self.grid)): # using subset of pixels

                    pix = self.grid[i]
                    weight = self.skymap1[pix]

                    if weight == 0:
                        continue
        
                    psr = PointSourceResponse(self.psr_axes[1:], f['hist/contents'][pix+1], unit = f['hist'].attrs['unit'])
                    pix_expectation = psr.get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi'])
        
                    total_expectation += pix_expectation*(weight*4*np.pi/self.skymap1.npix)

            # Add source to signal and update source counter:
            if self.src_counter == 0:
                self._signal = total_expectation
            if self.src_counter != 0:
                self._signal += total_expectation
            self.src_counter += 1

        # Get expectation for point sources:
        for name,source in point_sources.items():

            if self._source is None:
                self._source = copy.deepcopy(source) # to avoid same memory issue
                     
            # Compute point source response (for fixed position)
            if self._psr is None:
            
                coord = self._source.position.sky_coord
            
                dwell_time_map = self._get_dwell_time_map(coord)
            
                self._psr = self._dr.get_point_source_response(dwell_time_map)
            
            # Option to also fit the position:
            elif (source.position.sky_coord != self._source.position.sky_coord):
                
                coord = source.position.sky_coord
                
                dwell_time_map = self._get_dwell_time_map(coord)
                
                self._psr = self._dr.get_point_source_response(dwell_time_map)
                
            # Caching source to self._source after position judgment
            if self._source is not None:
                self._source = copy.deepcopy(source)

            # Convolve with spectrum
            spectrum = source.spectrum.main.shape
            self._signal = self._psr.get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi'])
            
        # Cache
        self._model = model

    def get_log_like(self):
        
        """
        Return the value of the log-likelihood
        """
        
        # Recompute the expectation if any parameter in the model changed
        if self._model is None:
            log.error("You need to set the model first")
        self.set_model(self._model)
        
        # Compute expectation including free background parameter.
        # Note: Need to check if self._signal is dense (i.e. np.ndarray) or sparse (i.e. sparse._coo.core.COO).
        # Currently using a quick workaround, but need a better method. 
        if self._fit_nuisance_params:
            if self.src_counter != 0:
                expectation = self._signal.contents + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents.todense()
            if self.src_counter == 0:
                expectation = self._signal.contents.todense() + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents.todense()
        
        # Compute expectation without background parameter
        else:
            if self.src_counter != 0:
                expectation = self._signal.contents + self._bkg.contents.todense()

            if self.src_counter == 0:    
                expectation = self._signal.contents.todense() + self._bkg.contents.todense()
        
        # Convert data into an arrary:
        data = self._data.contents 
        
        # Compute the log-likelihood
        log_like = np.nansum(data*np.log(expectation) - expectation)
        
        # Need to mask zero-values pixels if obtaining infinite likelihood.
        # Note: the mask function gives errors sometimes. This is a bug that needs to be fixed. 
        if log_like == -np.inf:
            logger.warning(f"There are bins in which the total expected counts = 0 but data != 0, making log-likelihood = -inf. "
                           f"Masking these bins.")
            log_like = np.nansum(np.ma.masked_invalid(data*np.log(expectation) - expectation))
        
        return log_like
    
    def inner_fit(self):
        
        """
        This fits nuisance parameters.
        """
        
        return self.get_log_like()
    
    def _get_dwell_time_map(self, coord):
        
        """Get the dwell time map of the source in the spacecraft frame.
        
        Parameters
        ----------
        coord: astropy.coordinates.SkyCoord; the coordinate of the target source.
        
        Returns
        -------
        dwell_time_map: mhealpy.containers.healpix_map.HealpixMap
        """
        
        self._sc_orientation.get_target_in_sc_frame(target_name = self._name, target_coord = coord)
        dwell_time_map = self._sc_orientation.get_dwell_map(response = self._rsp_path)
        
        return dwell_time_map
 
    def set_inner_minimization(self, flag: bool):
        
        """
        Turn on the minimization of the internal COSI parameters
        :param flag: turning on and off the minimization  of the internal parameters
        :type flag: bool
        :returns:
        """
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params
