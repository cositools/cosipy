from histpy import Histogram#, Axes, Axis

import numpy as np
import astropy.units as u
#from astropy.units import Quantity
#from scipy import integrate
from scoords import SpacecraftFrame, Attitude

#from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian

from .functions import get_integrated_spectral_model

import logging
logger = logging.getLogger(__name__)

class PointSourceResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the expected
    response of the instrument for a particular point in the sky.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy. Optional
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area convolved with wht source exposure.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical units, if not specified as part of ``contents``. Units of ``area*time``
        are expected.
    """
    
    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']
       
    def get_expectation(self, spectrum, polarization_level=None, polarization_angle=None, scatt_map=None, convention=None):
        """
        Convolve the response with a spectral (and optionally, polarization) hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.
        polarization_level : float, optional
            Polarization level (between 0 and 1).
        polarization_angle : :py:class:`cosipy.polarization.polarization_angle.PolarizationAngle`, optional
            Polarization angle. If in the spacecraft frame, the angle must have the same convention as the response.
        scatt_map : :py:class:`cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap`, optional
            Spacecraft attitude map. Only needed for polarization angle provided in galactic reference frame.
        convention : str, optional
            Polarization angle convention of response ('RelativeX', 'RelativeY', or 'RelativeZ'). Only needed for polarization angle provided in galactic reference frame.

        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """

        if polarization_level == None:

            if polarization_angle == None:

                if 'Pol' in self.axes.labels:

                    raise RuntimeError("Must include polarization in point source response if using polarization response")

                else:

                    contents = self.contents
                    axes = self.axes[1:]

            else:

                raise RuntimeError("Must provide both polarization level and angle")

        elif polarization_angle != None:

            from cosipy.polarization.polarization_angle import PolarizationAngle
            from cosipy.polarization.conventions import MEGAlibRelativeX, MEGAlibRelativeY, MEGAlibRelativeZ, IAUPolarizationConvention

            if polarization_level > 1.0 or polarization_level < 0:

                raise RuntimeError("Polarization level must be a fraction between 0 and 1")

            elif not 'Pol' in self.axes.labels:
                
                raise RuntimeError("Response must have polarization angle axis to include polarization in point source response")

            elif type(polarization_angle.convention.frame) == SpacecraftFrame:

                logger.warning("The response must have the same polarization convention as the provided polarization angle, or the result will be incorrect!")

                if polarization_angle.angle.deg == 180.:
                    polarization_angle = PolarizationAngle(0. * u.deg, polarization_angle.skycoord, convention=polarization_angle.convention)

                polarization_angle_components = []

                for i in range(self.axes['Pol'].nbins):

                    polarization_angle_components.append(self.slice[{'Pol':slice(i,i+1)}].project('Ei', 'Em', 'Phi', 'PsiChi'))

                    if polarization_angle.angle.deg >= self.axes['Pol'].edges.to_value(u.deg)[i] and polarization_angle.angle.deg < self.axes['Pol'].edges.to_value(u.deg)[i+1]:
                        polarized_component = polarization_angle_components[i].contents

                unpolarized_component = polarization_angle_components[0].contents

                for i in range(len(polarization_angle_components) - 1):

                    unpolarized_component += polarization_angle_components[i+1].contents

                polarized_component /= np.sum(polarized_component.value)
                unpolarized_component /= np.sum(unpolarized_component.value)

                polarization_hist = (polarization_level * polarized_component) + ((1 - polarization_level) * unpolarized_component)
                polarization_hist *= np.sum(self.contents) / np.sum(polarization_hist)

                contents = polarization_hist
                axes = self.project('Ei', 'Em', 'Phi', 'PsiChi').axes[1:]

            elif type(scatt_map) != None and convention != None:

                polarization_angle_components = np.empty(self.axes['Pol'].nbins, dtype=Histogram)

                if polarization_angle.angle.deg == 180.:
                    polarization_angle = PolarizationAngle(0. * u.deg, polarization_angle.skycoord, convention=polarization_angle.convention)

                for i in range(self.axes['Pol'].nbins):

                    response_slice = self.slice[{'Pol':slice(i,i+1)}].project('Ei', 'Em', 'Phi', 'PsiChi')

                    for j, (pixels, exposure) in enumerate(zip(scatt_map.contents.coords.transpose(), scatt_map.contents.data)):

                        attitude = Attitude.from_axes(x=scatt_map.axes['x'].pix2skycoord(pixels[0]), y=scatt_map.axes['y'].pix2skycoord(pixels[1]))

                        if convention == 'RelativeX':
                            this_convention = MEGAlibRelativeX(attitude=attitude)
                        elif convention == 'RelativeY':
                            this_convention = MEGAlibRelativeY(attitude=attitude)
                        elif convention == 'RelativeZ':
                            this_convention = MEGAlibRelativeZ(attitude=attitude)
                        else:
                            raise RuntimeError("Response convention must be 'RelativeX', 'RelativeY', or 'RelativeZ'")

                        polarization_angle_galactic = PolarizationAngle(self.axes['Pol'].centers.to_value(u.deg)[i] * u.deg, polarization_angle.skycoord, convention=this_convention).transform_to(polarization_angle.convention)

                        if polarization_angle_galactic.angle.deg == 180.:
                            polarization_angle_galactic = PolarizationAngle(0. * u.deg, polarization_angle_galactic.skycoord, convention=polarization_angle_galactic.convention)

                        polarization_angle_index = np.max(np.where(polarization_angle_galactic.angle.deg >= self.axes['Pol'].edges.to_value(u.deg)))

                        if hasattr(polarization_angle_components[polarization_angle_index], 'axes'):
                            polarization_angle_components[polarization_angle_index] += response_slice * exposure / np.sum(scatt_map.project('x'))
                        else:
                            polarization_angle_components[polarization_angle_index] = response_slice * exposure / np.sum(scatt_map.project('x'))

                for i in range(self.axes['Pol'].nbins):

                    if polarization_angle.angle.deg >= self.axes['Pol'].edges.to_value(u.deg)[i] and polarization_angle.angle.deg < self.axes['Pol'].edges.to_value(u.deg)[i+1]:
                        polarized_component = polarization_angle_components[i].contents

                unpolarized_component = polarization_angle_components[0].contents

                for i in range(len(polarization_angle_components) - 1):

                    unpolarized_component += polarization_angle_components[i+1].contents

                polarized_component /= np.sum(polarized_component.value)
                unpolarized_component /= np.sum(unpolarized_component.value)

                polarization_hist = (polarization_level * polarized_component) + ((1 - polarization_level) * unpolarized_component)
                polarization_hist *= np.sum(self.contents) / np.sum(polarization_hist)

                contents = polarization_hist
                axes = self.project('Ei', 'Em', 'Phi', 'PsiChi').axes[1:]

            else:

                raise RuntimeError("Scatt map and response convention must be provided to include polarization in galactic coordinates in point source response")

        else:

            raise RuntimeError("Must provide both polarization level and angle to include polarization in point source response")

        energy_axis = self.photon_energy_axis

        flux = get_integrated_spectral_model(spectrum, energy_axis)
        
        expectation = np.tensordot(contents, flux.contents, axes=([0], [0]))
        
        # Note: np.tensordot loses unit if we use a sparse matrix as it input.
        if self.is_sparse:
            expectation *= self.unit * flux.unit

        hist = Histogram(axes, contents=expectation)

        return hist
