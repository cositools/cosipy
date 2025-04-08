import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import itertools
from copy import deepcopy

import numpy as np
import astropy.units as u
from histpy import Histogram
import mhealpy as hp

class DetectorResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the
    response of the instrument for a particular :py:class:`.SpacecraftFrame` coordinate
    location.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical area units, if not specified as part of ``contents``
    """

    def __init__(self, coord, **kwargs):

        super().__init__(**kwargs)
        
        self.coord = coord
        self._set_mapping()

        self._spec = None
        self._aeff = None

    def drop_empty_axes(self):
 
         axis_names = self.axes.labels
         axes_projection = []
 
         for name in axis_names:
             if (name in ['Ei', 'Em', 'eps', 'Eps']) | (self.axes[name].edges.size > 2):
                 axes_projection.append(name)
 
         spec = self.project(axes_projection)
 
         return DetectorResponse(edges = spec.axes,
                                 contents = spec.contents,
                                 unit = spec.unit,
                                 interpolated_NuLambda = self.interpolated_NuLambda)

    def _set_mapping(self):
        self.mapping = {}
        target_names = ['Ei', 'Em', 'Phi', 'PsiChi']
        numlabels = len(self.axes.labels)

        for key, label in zip(target_names, self.axes.labels):
            self.mapping[key] = label           # Format: key_target : label
                                                # Example: {'Ei': 'Ei', 'Em': 'eps', 'Phi': 'Phi', 'PsiChi': 'PsiChi'}

    def _get_all_interp_weights(self, target: dict):

        indices = []
        weights = []

        for key in target:
            label = self.mapping[key]
            axis = self.axes[label]
            axis_scale = axis._scale
            axis_type = str(type(axis)).split('.')[-1].strip("'>")      # XXX: Could probably be simplified using `isinstance()`

            # Scale
            if axis_scale in ['linear', 'log']:   # To ensure nonlinear binning parametrizations are converted to a linear scale.

                # Axis Type
                if axis_type in ['Axis', 'HealpixAxis']:
                    if key == label:    # If key and label are the same, then there was no reparametrization along this axis
                        idx, w = axis.interp_weights(target[key])
                    else:
                        centers = self.transform_Eps_to_Em(axis.centers, target['Ei'])      # Transform coordinates to more physical units      # TODO: Generalize this
                        absdiff = np.abs(centers - target[key])                             # Calculate absolute difference to given target
                        idx = np.argpartition(absdiff, (1,2))[:2]                               # Find indices corresponding to two smallest absdiff
                        w = 1 - np.partition(absdiff, (1,2))[:2] / (centers[1] - centers[0])    # Calculate weights corresponding to two smallest absdiff
                else:
                    raise ValueError(f'Axis type: {axis_type} is not supported')
                
            # elif axis_scale == 'nonlinear':   
            #     pass

            else:
                raise ValueError(f'{axis_scale} binning / scale scheme is not supported')
            
            indices.append(idx)
            weights.append(w)
        
        return (indices, weights)
    
    def transform_Eps_to_Em(self, eps, Ei0):
        return (eps + 1) * Ei0

    def transform_Em_to_Eps(self, Em, Ei0):
        return Em/Ei0 - 1
    
    def transform_PhiPsiChi_to_ThetaZeta(self, Phi, PsiChi):        # Phi in degrees, PsiChi in pix
        nside_det = self.axes['SigmaTau'].nside

        source_vec = np.array(hp.ang2vec(self.coord.lon.deg, self.coord.lat.deg, lonlat=True)).reshape(3, -1)
        scatter_vec = np.array(hp.pix2vec(nside_det, PsiChi)).reshape(3,-1)

        Phi_geo = np.rad2deg(np.arccos(np.sum(source_vec * scatter_vec, axis=0)))
        Theta = Phi_geo - Phi

        xaxis = np.array([1., 0., 0.])      # Polarization angle basis
        pz = -source_vec.T
        px = np.cross(pz, xaxis)
        px /= np.linalg.norm(px, axis = 1).reshape(-1,1)
        py = np.cross(pz, px)
        py /= np.linalg.norm(py, axis = 1).reshape(-1,1)
        
        proj_scatter_vec = np.cross(pz, scatter_vec.T)
        change_basis = np.stack((px, py, pz), axis=1)
        proj_scatter_vec = np.matmul(change_basis, proj_scatter_vec[..., None])[..., 0]
        
        Zeta = np.rad2deg(np.arctan2(proj_scatter_vec[:, 1], proj_scatter_vec[:, 0])) - 90
        Zeta = np.where(Zeta < 0, Zeta + 360, Zeta)

        return Theta, Zeta

    # def transform_ThetaZeta_to_PhiPsiChi(self, Theta, Zeta):
    #     nside_det = self.axes['SigmaTau'].nside
        
    #     source_vec = np.array(hp.ang2vec(self.coord.lon.deg, self.coord.lat.deg, lonlat=True)).reshape(3, -1)
        
    #     Zeta = np.deg2rad(Zeta + 90) % 360
        
    #     xaxis = np.array([1., 0., 0.]) 
    #     pz = -source_vec.T
    #     px = np.cross(pz, xaxis)
    #     px /= np.linalg.norm(px, axis = 1).reshape(-1,1)
    #     py = np.cross(pz, px)
    #     py /= np.linalg.norm(py, axis = 1).reshape(-1,1)
        
    #     proj_scatter_vec = np.stack((np.cos(Zeta), np.sin(Zeta), np.zeros_like(Zeta)), axis=-1)
    #     change_basis = np.linalg.inv(np.stack((px, py, pz), axis=1))
    #     scatter_vec = np.matmul(change_basis, proj_scatter_vec[..., None])[..., 0]
        
    #     Phi_geo = np.rad2deg(np.arccos(np.dot(scatter_vec, source_vec)))
    #     PsiChi = hp.vec2pix(nside_det, scatter_vec[:, 0], scatter_vec[:, 1], scatter_vec[:, 2])
        
    #     Phi = Phi_geo - Theta

    #     return Phi, PsiChi

    def get_nearest_neighbors(self, target: dict, indices=None):
        if indices is not None:
            neighbors = {}
            for idx, key in zip(indices, target):
                label = self.mapping[key]
                neighbors[label] = self.axes[label].centers[idx]
            return neighbors
        
        else:
            target = dict(sorted(target.items()))
            indices, _ = self._get_all_interp_weights(target)
            return self.get_nearest_neighbors(target, indices)

    def get_interp_response(self, target: dict):
        """
        Currently only supports nonlinear spectral responses (
        and for a particular parametrization)
        TODO: In the future, this will also support nonlinear / 
        piecewise-linear directional responses.
        """

        # target = dict(sorted(target.items()))       # Sort dictionary by key (XXX: assuming response matrix also sorts in the same way)
        indices, weights = self._get_all_interp_weights(target)
        perm_indices = list(itertools.product(*indices))
        perm_weights = list(itertools.product(*weights))

        if len(target) == len(self.axes):
            interpolated_response_value = 0
            for idx, w in zip(perm_indices, perm_weights):
                interpolated_response_value += np.prod(w) * self.contents[idx]

        else:
            interpolated_response_value = np.zeros(len(self.axes['Ei']) - 1) * self.contents.unit     # XXX: Assuming all measured variables require interpolation
            for idx, w in zip(perm_indices, perm_weights):
                i = (Ellipsis,) + idx   # XXX: Assuming 'Ei' is the first index
                interpolated_response_value += np.prod(w) * self.contents[i]

        self.neighbors = self.get_nearest_neighbors(target, indices)
        
        return interpolated_response_value
    
    def get_spectral_response(self, copy = True):
        """
        Reduced detector response, projected along the real and measured energy axes only.
        The Compton Data Space axes are not included.

        Parameters
        ----------
        copy : bool
            If true, a copy of the cached spectral response will be returned.
        
        Returns
        -------
        :py:class:`DetectorResponse`
        """

        # Cache the spectral response
        if self._spec is None:
            spec = self.project(['Ei', self.mapping['Em']])
            self._spec = DetectorResponse(coord = self.coord,
                                          edges = spec.axes,
                                          contents = spec.contents,
                                          unit = spec.unit)

        if copy:
            return deepcopy(self._spec)
        else:
            return self._spec
        
    def get_effective_area(self, energy = None, copy = True):
        """
        Compute the effective area at a given energy. If no energy is specified, the
        output is a histogram for the effective area at each energy bin.

        Parameters
        ----------
        energy : optional, :py:class:`astropy.units.Quantity`
            Energy/energies at which to interpolate the linearly effective area
        copy : bool
            If true, a copy of the cached effective will be returned.
        
        Returns
        -------
        :py:class:`astropy.units.Quantity` or :py:class:`histpy.Histogram`
        """
        
        if self._aeff is None:
            self._aeff = self.get_spectral_response(copy = False).project('Ei').to_dense()

        if energy is None:
            if copy:
                return deepcopy(self._aeff)
            else:
                return self._aeff
        else:
            return self._aeff.interp(energy)

    def get_dispersion_matrix(self):
        """
        Compute the energy dispersion matrix, also known as migration matrix. This holds the
        probability of an event with real energy ``Ei`` to be reconstructed with an measured
        energy ``Em``.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """
        
        # Get spectral response and effective area normalization
        spec = self.get_spectral_response(copy = False)
        norm = self.get_effective_area().full_contents

        # Hack the under/overflow bins to supress 0/0 wearning
        norm[0] = 1*norm.unit if norm[0] == 0 else norm[0]
        norm[-1] = 1*norm.unit if norm[-1] == 0 else norm[-1]

        # Avoid another 0/0 is the effective area is null for some bins
        if np.any(norm == 0):
            norm[norm == 0] = 1*norm.unit

            logger.warn("Null effective area, cannot properly compute dispersion matrix.")
        
        # "Broadcast" such that it has the compatible dimensions with the 2D matrix
        norm = spec.expand_dims(norm, 'Ei')
        
        # Normalize column-by-column
        return (spec / norm)
    
        # NOTE: When histpy is updated to do away without overflow and
        # underflow bins, the following lines of code can replace this
        # function body.

        # # Get spectral response and effective area normalization
        # spec = self.get_spectral_response(copy = False)
        # norm = self.get_effective_area().contents

        # # "Broadcast" such that it has the compatible dimensions with the 2D matrix
        # norm = spec.expand_dims(norm, 'Ei')
        
        # # Normalize column-by-column
        # return (spec / norm)    # XXX: Runtime warning needs to be dealt with in histpy. 
        #                         # Overflow and underflow bins functionality should be removed.

    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']

    
    @property
    def measured_energy_axis(self):
        """
        Measured energy bins (``Em``).

        Returns
        -------
        :py:class:`histpy.Axes`        
        """
        
        return self.axes[self.mapping['Em']]
        

        
    
