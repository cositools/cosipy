from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from astropy import units as u

from cosipy.polarization import PolarizationConvention

class RelativeCDSCoordinates:

    def __init__(self,
                 source_direction:Union[SkyCoord, np.ndarray[float]],
                 pol_convention:PolarizationConvention):
        """
        Size N

        Parameters
        ----------
        source_direction: SkyCoord or normalized vector (3,N)
            If a SkyCoord, it is transformed to ``pol_convention.frame``
            before computing the local polarization basis, so any frame
            is accepted.
            If a bare vector, NO frame transform is performed -- it is
            taken as-is and assumed to already be expressed in
            ``pol_convention.frame``.
        pol_convention
        """

        if isinstance(source_direction, SkyCoord):

            # Convert to convention frame
            self._frame = pol_convention.frame
            self._representation_type = source_direction.representation_type
            source_direction = source_direction.transform_to(self._frame)
            self._source_vec = self._standardize_vector(source_direction)

        else:

            # Assume it's already in the convention frame
            self._frame = None
            self._representation_type = None
            self._source_vec = source_direction

        self._px, self._py = pol_convention.get_basis_local(self._source_vec)

    @staticmethod
    def _standardize_angle(angle):
        if isinstance(angle, (Quantity, Angle)):
            angle = angle.to_value(u.rad)

        return np.asarray(angle)

    @staticmethod
    def _standardize_vector(direction):
        if isinstance(direction, SkyCoord):
            direction = direction.cartesian.xyz

        return np.asarray(direction)

    def to_cds(self, phi, az):
        """From coordinate relative to the source direction to the gamma-ray
        scattered direction.

        Parameters
        ----------
        phi:
           Angular distance with respect to the source direction. Can
           have shape (N,) or (N,M).
        az:
           Azimuthal angle around the source direction, with a
           0-direction defined by the polarization convention. Same
           size as phi or broadcastable.

        Returns
        -------
        The scattered direction
        Shape:
        If working with pure vectors: (3, N, M) (or broadcastable, e.g. (3,1,M)
        If working with SkyCoord: (N, M)

        """

        # 1. Convert to a numpy array of radians
        # 2. Add axis to broadcast with x,y,z coordinates
        phi = self._standardize_angle(phi)
        az = self._standardize_angle(az)

        # Get the right shape for broadcasting
        phi,az = np.broadcast_arrays(phi, az)
        phi = phi[np.newaxis]
        az = az[np.newaxis]
        new_dims = tuple(range(self._source_vec.ndim, phi.ndim))
        source_vec = np.expand_dims(self._source_vec, new_dims)
        px = np.expand_dims(self._px, new_dims)
        py = np.expand_dims(self._py, new_dims)

        # Sum over each basis vector, without allocating multiple arrays
        psichi_vec = px * np.cos(az)
        psichi_vec += py * np.sin(az)
        psichi_vec *= np.sin(phi)
        psichi_vec += source_vec * np.cos(phi)


        # Convert to skycoord if needed
        if self._frame is not None:

            psichi = SkyCoord(*psichi_vec,
                              representation_type='cartesian',
                              frame=self._frame)

            psichi.representation_type = self._representation_type

            return psichi

        else:

            return psichi_vec

    def to_relative(self, psichi:Union[SkyCoord, np.ndarray[float]]):
        """From the absolute scattered direction, to the coordinates relative
        to the source direction.

        Parameters
        ----------
        psichi:
        Scattered direction
        Can have shape:
        - Vector: (3,N) or (3,N,M) (or broadcastable, e.g. (3,1,M)
        - Skycoord: (N,) or (N,M).

        Returns
        -------
        phi,az:
        phi: Angular distance with respect to the source direction.
            Range [0, pi] .
        az: Azimuthal angle around the source direction, with a
            0-direction defined by the polarization convention. Range is (-pi, pi]
        Each with shape (N,M). Angles.

        """

        psichi_vec = self._standardize_vector(psichi)

        # Adjust dimensions for broadcasting
        new_dims = tuple(range(self._source_vec.ndim, psichi_vec.ndim))
        source_vec = np.expand_dims(self._source_vec, new_dims)
        px = np.expand_dims(self._px, new_dims)
        py = np.expand_dims(self._py, new_dims)

        # Get the psichi_perp_vec component along each basis vector
        # This is equivalent to
        # psichi_px_component = np.sum(px * psichi_perp_vec, axis=0)
        # for each component
        # but it does not allocate the temporary px*psichi_perp_vec results
        # and performs the full operation in one step
        psichi_px_component, psichi_py_component, psichi_source_component =  \
            np.einsum('ji...,ji...->j...',[px,py,source_vec], psichi_vec[np.newaxis])

        # Get the angle from the vector
        phi = np.arccos(psichi_source_component)
        az = np.arctan2(psichi_py_component, psichi_px_component)

        return Angle(phi, unit=u.rad, copy=False), Angle(az, unit=u.rad, copy=False)

    @staticmethod
    def get_relative_cds_phase_space(phi_min = None, phi_max = None,
                                     arm_min = None, arm_max = None,
                                     az_min = None, az_max = None):
        """
        The CDS is described by:
        phi: the polar scattering angle
        psichi: the direction of the scattered gamma

        Given a source direction, psichi can be parametrized with
        - arm equals the minimum angular distance between the psichi
          and a cone centered at the source direction with hald-opening
          angle equal to phi
        - az: the azimuthal angle around the source direction

        The total phase space of psichi is that of the sphere. If psi
        is the colatitude and chi the longitude, then
        dV = sin(psi) dphi dpsi dchi

        The total phase space is pi (from phi) time 4*pi (from psichi,
        that is the sphere area)

        In the reparametrization, this is
        dV = sin(phi + arm) dphi darm daz

        While the total phase space remains unchanged, in order to
        integrate this volume in arbitrary limits you need take into
        account the fact that phi+arm range is limited to [0,pi].

        This function performs such integration by checking all
        possible integration limit cases.

        Parameters
        ----------
        phi_min: Defaults to 0
        phi_max: default to pi
        arm_min: default to -pi
        arm_max: default to pi
        az_min: default to 0
        az_max: default to 2*pi

        Returns
        -------
        Phase space

        """

        if phi_min is None:
            phi_min = 0

        if phi_max is None:
            phi_max = np.pi

        if arm_min is None:
            arm_min = -np.pi

        if arm_max is None:
            arm_max = np.pi

        if az_min is None:
            az_min = 0

        if az_max is None:
            az_max = 2*np.pi

        phi_min = RelativeCDSCoordinates._standardize_angle(phi_min)
        phi_max = RelativeCDSCoordinates._standardize_angle(phi_max)
        arm_min = RelativeCDSCoordinates._standardize_angle(arm_min)
        arm_max = RelativeCDSCoordinates._standardize_angle(arm_max)
        az_min = RelativeCDSCoordinates._standardize_angle(az_min)
        az_max = RelativeCDSCoordinates._standardize_angle(az_max)

        phi_min, phi_max, arm_min, arm_max, az_min, az_max = np.broadcast_arrays(phi_min, phi_max, arm_min, arm_max, az_min, az_max)

        # Handle cases in between the physical boundaries
        # Integrate excluding unphysical corners
        # Remove unphysical rectangles
        arm_min = np.choose((arm_min < -phi_max) & (-phi_max < arm_max), [arm_min, -phi_max])
        arm_max = np.choose((arm_min < np.pi - phi_min) & (np.pi - phi_min < arm_max), [arm_max, np.pi - phi_min])

        phi_min = np.choose((phi_min < -arm_max) & (-arm_max < phi_max), [phi_min, -arm_max])
        phi_max = np.choose((phi_min < np.pi - arm_min) & (np.pi - arm_min < phi_max), [phi_max, np.pi - arm_min])

        integral_rect = (az_max - az_min) * (
                -np.sin(arm_min + phi_min) + np.sin(arm_max + phi_min) + np.sin(arm_min + phi_max) - np.sin(arm_max + phi_max))

        # Remove unphysical corners (triangles or trapezoids)
        # Note the (phi1 + arm1) and (phi2 + arm2) masks in front

        # Lower left corner (low phi, low arm)
        # Integrate[Sin[phi+arm],{phi,phi1,phi2},{arm,arm1, -phi}]
        #     //FullSimplify
        phil = np.maximum(-arm_max, phi_min)
        phih = np.minimum(-arm_min, phi_max)
        unphys_lowerleft_integral = -phih + phil + np.sin(arm_min + phih) - np.sin(arm_min + phil)
        unphys_lowerleft_integral *= (phil + arm_min < 0)
        integral = integral_rect - (az_max - az_min) * unphys_lowerleft_integral

        # Upper right corner (high phi, high arm)
        # Integrate[Sin[phi+arm],{phi,phi1,phi2}, {arm, \[Pi]-phi, arm2}]
        #     //FullSimplify
        phil = np.maximum(np.pi - arm_max, phi_min)
        phih = np.minimum(np.pi - arm_min, phi_max)
        unphys_upperright_integral = phil - phih + np.sin(arm_max + phil) - np.sin(arm_max + phih)
        unphys_upperright_integral *= (phih + arm_max > np.pi)
        integral -= (az_max - az_min) * unphys_upperright_integral

        # Handle fully physical or fully unphysical
        fully_phys = (phi_min + arm_min >= 0) & (phi_max + arm_max <= np.pi)
        fully_unphys = (phi_max + arm_max <= 0) | (phi_min + arm_min >= np.pi)

        # Mathematica: Integrate[Sin[phi+arm], {phi,phi1,phi2} , {arm,arm1,arm2}]//FullSimplify
        integral_full = (az_max - az_min) * (-np.sin(arm_min + phi_min) + np.sin(arm_max + phi_min) + np.sin(arm_min + phi_max) - np.sin(arm_max + phi_max))

        if integral.ndim == 0:
            if fully_phys:
                return integral
            if fully_unphys:
                return 0
        else:
            integral[fully_phys] = integral_full[fully_phys]
            integral[fully_unphys] = 0

        return integral
