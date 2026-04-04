from typing import Union, Optional

import histpy
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astropy.units import Quantity
from scoords import Attitude, SpacecraftFrame

from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.data_io import EmCDSBinnedData
from cosipy.interfaces import BinnedDataInterface
from cosipy.interfaces.instrument_response_interface import BinnedInstrumentResponseInterface

from cosipy.polarization import PolarizationAngle, PolarizationAxis
from cosipy.response import FullDetectorResponse

from histpy import Axes, Histogram


__all__ = ["BinnedInstrumentResponse"]

class BinnedInstrumentResponse(BinnedInstrumentResponseInterface):

    def __init__(self, response:FullDetectorResponse, data:EmCDSBinnedData):

        self._dr = response

        self._axes = data.axes

        if set(self._axes.labels) != {'Em','PsiChi','Phi'}:
            raise ValueError(f"Unexpected axes labels. Expecting \"{{'Em','PsiChi','Phi'}}\", got {axes.labels}")

        if self._dr.measurement_axes["Em"] != self._axes["Em"]:
            # Matches the v0.3 behaviour
            raise ValueError("This implementation can only handle a fixed measured energy (Em) binning equal to the underlying response file.")

        if self._dr.measurement_axes["Phi"] != self._axes["Phi"]:
            # Matches the v0.3 behaviour
            raise ValueError("This implementation can only handle a fixed scattering angle (Phi) binning equal to the underlying response file.")

        if self._axes["PsiChi"].coordsys is None:
            raise ValueError("PsiChi axes doesn't have a coordinate system")


    @property
    def axes(self) -> histpy.Axes:
        return self._axes

    @property
    def is_polarization_response(self):
        return 'Pol' in self._dr.axes.labels

    def differential_effective_area(self,
                                    direction: SkyCoord,
                                    energy:u.Quantity,
                                    polarization:PolarizationAxis = None,
                                    attitude:Attitude = None,
                                    time = None,
                                    weight:Union[Quantity, float] = None,
                                    out:Quantity = None,
                                    add_inplace:bool = False) -> Quantity:
        """
        Interpolations and bin coupling:
        * The direction is always bi-linearly interpolated.
        * Ei, Em and Phi always needs to match the response exactly
        * If PsiChi is in local coordinates, PsiChi and polarization need to match the response exactly
        * If PsiChi is in inertial coordinates, PsiChi and polarization are interpolated at 0-th order during the rotation

        Parameters
        ----------
        data
            Binned measurements. We can only handle EmCDSBinnedData
        direction:
            Photon incoming direction in SC coordinates
        energy:
            Photon energy
        polarization
            Photon polarization angle axis
        attitude
            Attitude defining the orientation of the SC in an inertial coordinate system.
        time:
            Ignored. Not time dependent.
        weight
            Optional. Weighting the result by a given weight. Providing the weight at this point as opposed to
            apply it to the output can result in greater efficiency.
        out:
            Optional. Histogram to store the output. If possible, the implementation should try to avoid allocating
            new memory.
        add_inplace
            If True and a Histogram output was provided, we will try to avoid allocating new
            memory and add --not set-- the result of this operation to the output.

        Returns
        -------
        The effective area times the event measurement probability distribution integrated on each of the bins
        of the provided axes
        """

        # Checks/limitations

        if not np.array_equal(energy, self._dr.axes['Ei'].centers):
            # Matches the v0.3 behaviour
            raise RuntimeError("Currently, the probed energy values need to match the underlying response matrix Ei centers.")

        if polarization is not None:
            if not self.is_polarization_response:
                raise RuntimeError("The FullDetectorResponse does not contain polarization information")

        if direction.shape != ():
            raise ValueError("Currently this implementation can only deal with one direction at a time")


        # Fork for local and galactic PsiChi coordinates
        if not isinstance(self.axes["PsiChi"].coordsys, SpacecraftFrame):
            # Is inertial

            if attitude is None:
                raise RuntimeError("User need to provide the attitude information in order to transform to spacecraft coordinates")

            return self._differential_effective_area_inertial(attitude, self.axes, direction, polarization, weight, out, add_inplace)

        # Is local

        # Check again remaining axes
        if self._dr.measurement_axes["PsiChi"] != self.axes["PsiChi"]:
            # Matches the v0.3 behaviour
            raise ValueError("This implementation can only handle a fixed scattering direction (PsiChi) binning equal to the underlying response file.")

        if polarization is not None:
            if not np.array_equal(polarization, self._dr.axes['Pol'].centers):
                # Matches the v0.3 behaviour
                raise RuntimeError(
                    "Currently, the probed polarization angles need to match the underlying response matrix Pol centers.")

        # Get the pixel as is since we already checked that the requested
        # energy and polarization points match the underlying response centers
        # Matches the v0.3 behaviour
        pix = self._dr.ang2pix(direction)

        # TODO: Update after Pr364. get_pixel(pix, weight) should make this more efficient
        if weight is not None:
            result = self._dr[pix] * weight
        else:
            result = self._dr[pix]

        # Fix order of output axes to the standard by the interface
        results_axes_labels = ['Ei']

        if polarization is not None:
            results_axes_labels += ['Pol']

        results_axes_labels += list(self.axes.labels)

        result = result.project(results_axes_labels)

        if polarization is None and self.is_polarization_response:
            # It was implicitly converted to unpolarized response by the
            # projection above, but this is still needed to get the mean
            result /= self._dr.axes.nbins

        return self._fill_out_and_return(result, out, add_inplace)

    @staticmethod
    def _fill_out_and_return(result:Histogram, out:Quantity, add_inplace:bool = False) -> Quantity:

        if out is None:
            # Convert to base class
            return result.contents
        else:

            if out.shape != result.shape:
                raise ValueError("The provided out argument doesn't have the right shape."
                                 f"Expected {result.shape}, got {out.axes.shape}")

            if add_inplace:
                out += result.contents
            else:
                out[:] = result.contents

            return out

    def _differential_effective_area_inertial(self,
                                              attitude:Attitude,
                                              axes:Axes,
                                              direction: SkyCoord,
                                              polarization:PolarizationAxis = None,
                                              weight:Union[float, Quantity] = None,
                                              out: Quantity = None,
                                              add_inplace:bool = False,
                                              ) -> Quantity:
        """
        Will rotate PsiChi from local to inertial coordinates

        Parameters
        ----------
        axes
        direction
        energy
        polarization
        attitude

        Returns
        -------

        """

        # Generate axes that will allow us to use _sum_rot_hist,
        # and obtain the same results as in v3.x
        out_axes = [self._dr.axes['Ei']]

        if self.is_polarization_response:

            #raise RuntimeError("Fix me. No pol yet")

            # Since we're doing a 0-th order interpolation, the only thing that matter are the bin centers,
            # so we're placing them at the input polarization angles

            if np.any(polarization.centers.angle.value[1:] - polarization.centers.angle.value[:-1] < 0):
                raise ValueError("This implementation requires strictly monotonically increasing polarization angles")

            pol_edges = (polarization.centers.angle[:-1] + polarization.centers.angle[1:]).to_value(u.deg)/2

            pol_edges = np.concatenate((np.array([pol_edges[0] - 2*(pol_edges[0] - polarization.centers.angle.to_value(u.deg)[0])]), pol_edges))
            pol_edges = np.concatenate((pol_edges, np.array([pol_edges[-1] + 2 * (polarization.centers.angle.to_value(u.deg)[-1] - pol_edges[-1])])))

            out_axes += [PolarizationAxis(pol_edges*u.deg, convention = polarization.convention)]

        out_axes += list(axes)
        out_axes = Axes(out_axes)

        if weight is None:
            # Weight takes the role of the exposure in _sum_rot_hist, which is not an optional argument
            weight = 1

        # Almost copy-paste from FullDetectorResponse.get_point_source_response(). Improve to avoid duplicated code
        def rotate_coords(c, rot):
            """
            Apply a rotation matrix to one or more 3D directions
            represented as Cartesian 3-vectors.  Return rotated directions
            in polar form as a pair (co-latitude, longitude) in
            radians.

            """
            c_local = rot @ c

            c_x, c_y, c_z = c_local

            theta = np.arctan2(c_y, c_x)
            phi = np.arccos(c_z)

            return (phi, theta)

        rot = attitude.transform_to('icrs').rot.inv().as_matrix()

        src_cart = direction.transform_to('icrs').cartesian.xyz.value
        loc_src_colat, loc_src_lon = rotate_coords(src_cart, rot)
        loc_src_pixels = self._dr._axes['NuLambda'].find_bin(theta=loc_src_colat,
                                                             phi=loc_src_lon)

        sf_psichi_axis = axes['PsiChi']
        sf_psichi_dirs = sf_psichi_axis.pix2skycoord(np.arange(sf_psichi_axis.nbins))
        sf_psichi_dirs_cart = sf_psichi_dirs.transform_to('icrs').cartesian.xyz.value
        loc_psichi_colat, loc_psichi_lon = rotate_coords(sf_psichi_dirs_cart, rot)
        loc_psichi_pixels = self._dr._axes['PsiChi'].find_bin(theta=loc_psichi_colat,
                                                              phi=loc_psichi_lon)


        # Either initialize a new or clear cache
        if out is None:
            out = Quantity(np.zeros(out_axes.shape), dr_pix.unit)
        else:
            if not add_inplace:
                out[:] = 0

        if isinstance(weight, u.Quantity):
            weight_unit = weight.unit
            weight = weight.value
        else:
            weight_unit = None

        out.value[:] += self._dr._rot_psr(out_axes, weight,
                                  loc_psichi_pixels,
                                  (loc_src_pixels,))

        if weight_unit is not None:
            out = u.Quantity(out.value, weight_unit*out.unit, copy = None)

        return out







