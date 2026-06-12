import numpy as np
import healpy as hp
from tqdm.autonotebook import tqdm
import sparse
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, cartesian_to_spherical, Galactic

from scoords import Attitude, SpacecraftFrame
from histpy import Histogram, Axes, Axis, HealpixAxis

from ..data_interfaces.utils import tensordot_sparse
from ..constants import EARTH_RADIUS_KM

class CoordsysConversionMatrix(Histogram):
    """
    A class for coordinate conversion matrix (ccm).
    """

    def __init__(self, edges, contents = None, sumw2 = None,
                 labels=None, axis_scale = None, sparse = None, unit = None,
                 binning_method = None, copy_contents = True):

        super().__init__(edges, contents = contents, sumw2 = sumw2,
                         labels = labels, axis_scale = axis_scale, sparse = sparse, unit = unit,
                         copy_contents = copy_contents)

        self.binning_method = binning_method #'Time' or 'ScAtt'

    def copy(self):
        new = super().copy()
        new.binning_method = self.binning_method
        return new

    @classmethod
    def from_exposure_table(cls, exposure_table, full_detector_response, nside_model = None, scheme_model = 'ring', use_averaged_pointing = False, earth_occ = True, r_earth = EARTH_RADIUS_KM):
        """
        Calculate a ccm from a given exposure_table.

        Parameters
        ----------
        full_detector_response : :py:class:`cosipy.response.FullDetectorResponse`
            Response
        exposure_table : :py:class:`cosipy.image_deconvolution.SpacecraftAttitudeExposureTable`
            Scatt exposure table
        nside_model : int or None, default None
            If it is None, it will be the same as the NSIDE in the response.
        scheme_model: str, default ring
        use_averaged_pointing : bool, default False
            If it is True, first the averaged Z- and X-pointings are calculated.
            Then the dwell time map is calculated once for ach model pixel and each scatt_binning_index.
            If it is False, the dwell time map is calculated for each attitude in zpointing and xpointing in the exposure table.
            Then the calculated dwell time maps are summed up.
            In the former case, the computation is fast but may lose the angular resolution.
            In the latter case, the conversion matrix is more accurate but it takes a long time to calculate it.
        earth_occ: bool, default True
            If it is True, the earth occultation is considered.
        r_earth : float, default EARTH_RADIUS_KM
            Earth's radius in kilometers.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix'
            Its axes are [ "ScAtt" or "Time", "lb", "NuLambda" ].
        """

        if nside_model is None:
            nside_model = full_detector_response.nside
        is_nest_model = True if scheme_model == 'nest' else False
        nside_local = full_detector_response.nside

        n_bins = len(exposure_table)

        axis_binning = Axis(edges = np.arange(n_bins+1), label = exposure_table.binning_method)
        axis_model_map = HealpixAxis(nside = nside_model, coordsys = "galactic", scheme = scheme_model, label = "lb")
        axis_local_map = full_detector_response.axes["NuLambda"]

        axis_coordsys_conv_matrix = Axes((axis_binning, axis_model_map, axis_local_map), copy_axes=False)

        contents = []

        for i_bin in tqdm(range(n_bins)):
            ccm_thispix = np.zeros((axis_model_map.nbins, axis_local_map.nbins)) # without unit

            row = exposure_table.iloc[i_bin]

            binning_index = row[exposure_table.binning_index_name]
            num_pointings = row['num_pointings']
            zpointing = row['zpointing']
            xpointing = row['xpointing']
            zpointing_averaged = row['zpointing_averaged']
            xpointing_averaged = row['xpointing_averaged']
            earth_zenith = row['earth_zenith']
            altitude = row['altitude']
            livetime = row['livetime']
            total_livetime = row['total_livetime']

            if use_averaged_pointing:
                z = SkyCoord([zpointing_averaged[0]], [zpointing_averaged[1]], frame="galactic", unit="deg")
                x = SkyCoord([xpointing_averaged[0]], [xpointing_averaged[1]], frame="galactic", unit="deg")
            else:
                z = SkyCoord(zpointing.T[0], zpointing.T[1], frame="galactic", unit="deg")
                x = SkyCoord(xpointing.T[0], xpointing.T[1], frame="galactic", unit="deg")

            attitude = Attitude.from_axes(x = x, z = z, frame = 'galactic')

            # exposure map calculation including earth occultation
            exposure_time_map = cls._calc_exposure_time_map(nside_model, num_pointings, earth_zenith, altitude, livetime,
                                                            is_nest_model = is_nest_model, earth_occ = earth_occ, r_earth = r_earth)

            # ccm
            for ipix in range(hp.nside2npix(nside_model)):
                l, b = hp.pix2ang(nside_model, ipix, nest=is_nest_model, lonlat=True)
                pixel_coord = SkyCoord(l, b, unit = "deg", frame = 'galactic')

                src_path_cartesian = SkyCoord(np.dot(attitude.rot.inv().as_matrix(), pixel_coord.cartesian.xyz.value),
                                              representation_type = 'cartesian', frame = SpacecraftFrame())

                src_path_spherical = cartesian_to_spherical(src_path_cartesian.x, src_path_cartesian.y, src_path_cartesian.z)

                l_scr_path = np.array(src_path_spherical[2].deg)  # note that 0 is Quanty, 1 is latitude and 2 is longitude and they are in rad not deg
                b_scr_path = np.array(src_path_spherical[1].deg)

                src_path_skycoord = SkyCoord(l_scr_path, b_scr_path, unit = "deg", frame = SpacecraftFrame())

                pixels, weights = axis_local_map.get_interp_weights(src_path_skycoord)

                if use_averaged_pointing:
                    weights = weights * np.sum(exposure_time_map[:,ipix])
                else:
                    weights = weights * exposure_time_map[:,ipix]

                hist, bins = np.histogram(pixels, bins = axis_local_map.edges, weights = weights)

                ccm_thispix[ipix] = hist

            ccm_thispix_sparse = sparse.COO.from_numpy( ccm_thispix.reshape((1, axis_model_map.nbins, axis_local_map.nbins)) )

            contents.append(ccm_thispix_sparse)

        coordsys_conv_matrix = cls(axis_coordsys_conv_matrix,
                                   contents = sparse.concatenate(contents),
                                   unit = u.s,
                                   copy_contents = False)

        coordsys_conv_matrix.binning_method = exposure_table.binning_method

        return coordsys_conv_matrix

    @classmethod
    def _calc_exposure_time_map(cls, nside_model, num_pointings, earth_zenith, altitude, livetime, is_nest_model, earth_occ, r_earth):
        """
        Calculate exposure time map considering Earth occultation.

        This method computes an exposure time map for each pointing, identifying
        pixels that are occulted by the Earth and assigning exposure times accordingly.
        For each pointing, pixels within the Earth's angular radius are identified
        and assigned the corresponding time interval.

        Parameters
        ----------
        nside_model : int
            HEALPix NSIDE parameter for the model map resolution.
        num_pointings : int
            Number of spacecraft pointings.
        earth_zenith : numpy.ndarray
            Array of shape (num_pointings, 2) containing the direction to Earth's center
            in galactic coordinates [longitude, latitude] in degrees for each pointing.
        altitude : numpy.ndarray
            Array of spacecraft altitudes in kilometers for each pointing.
        livetime : numpy.ndarray
            Array of livetimes in seconds for each pointing.
        is_nest_model : bool
            If True, use nested HEALPix pixel ordering scheme. If False, use ring ordering.
        earth_occ: bool
            If it is True, the earth occultation is considered.
        r_earth : float
            Earth's radius in kilometers.

        Returns
        -------
        numpy.ndarray
            Exposure time map of shape (num_pointings, npix_model), where npix_model
            is the total number of HEALPix pixels. Each element [i, j] contains the
            exposure time in seconds for pointing i and pixel j that is within the
            Earth occultation region.
        """

        npix_model = hp.nside2npix(nside_model)

        exposure_time_map = np.zeros((num_pointings, npix_model))

        for i_pointing in range(num_pointings):
            if earth_occ:
                earth_radius = np.pi - np.arcsin(r_earth / (r_earth + altitude[i_pointing])) #rad
                filling_pixel_index = hp.query_disc(nside_model, hp.ang2vec(earth_zenith[i_pointing,0], earth_zenith[i_pointing,1], lonlat = True), nest = is_nest_model, radius = earth_radius)
                exposure_time_map[i_pointing][filling_pixel_index] = livetime[i_pointing]
            else:
                exposure_time_map[i_pointing, :] = livetime[i_pointing]

        return exposure_time_map

    @classmethod
    def open(cls, filename, name = 'hist'):
        """
        Open a ccm from a file.

        Parameters
        ----------
        filename : str
            Path to file.
        name : str, default 'hist'
            Name of group where the histogram was saved.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix'
            Its axes are [ "lb", "Time" or "ScAtt", "NuLambda" ].
        """

        new = super().open(filename, name)

        new.binning_method = new.axes.labels[0] # 'Time' or 'ScAtt'

        return new

    def calc_exposure_map(self, full_detector_response):
        """
        Calculate the exposure map from the coordinate conversion matrix and detector response.

        Performs a tensor dot product between the CCM and the effective area, contracting
        over the 'NuLambda' axis to transform from local spacecraft coordinates to sky coordinates.

        Parameters
        ----------
        full_detector_response : :py:class:`cosipy.response.FullDetectorResponse`
            Full detector response

        Returns
        -------
        :py:class:`histpy.Histogram`
            Exposure map with axes ["ScAtt", "lb", "Ei"] representing the effective area x time
            for each attitude bin, sky pixel, and energy bin.
        """

        effective_area = full_detector_response.to_dr().project(['NuLambda', 'Ei'])

        exposure_map_contents = tensordot_sparse(self.contents, self.unit,
                                                 effective_area.contents, axes = ([2],[0]))
        # ["ScAtt", "lb", "NuLambda"] x ["NuLambda", "Ei"]
        exposure_map_axes = [self.axes[self.binning_method], self.axes['lb'], effective_area.axes['Ei']]

        exposure_map = Histogram(exposure_map_axes, exposure_map_contents)

        return exposure_map
