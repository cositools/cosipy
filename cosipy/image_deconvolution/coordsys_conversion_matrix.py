import numpy as np
import healpy as hp
from tqdm.autonotebook import tqdm
import sparse
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, cartesian_to_spherical, Galactic

from scoords import Attitude, SpacecraftFrame
from histpy import Histogram, Axes, Axis, HealpixAxis

class CoordsysConversionMatrix(Histogram):
    """
    A class for coordinate conversion matrix (ccm).
    """

    def __init__(self, edges, contents = None, sumw2 = None,
                 labels=None, axis_scale = None, sparse = None, unit = None,
                 binning_method = None):
        
        super().__init__(edges, contents = contents, sumw2 = sumw2,
                         labels = labels, axis_scale = axis_scale, sparse = sparse, unit = unit)

        self.binning_method = binning_method #'Time' or 'ScAtt'

    @classmethod
    def time_binning_ccm(cls, full_detector_response, orientation, time_intervals, nside_model = None, is_nest_model = False):
        """
        Calculate a ccm from a given orientation.
        
        Parameters
        ----------
        full_detector_response : :py:class:`cosipy.response.FullDetectorResponse`
            Response
        orientation : :py:class:`cosipy.spacecraftfile.SpacecraftFile`
            Orientation
        time_intervals : :py:class:`np.array`
            The same format of binned_data.axes['Time'].edges
        nside_model : int or None, default None
            If it is None, it will be the same as the NSIDE in the response.
        is_nest_model : bool, default False
            If scheme of the model map is nested, it should be False while it is rare.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix`
            Its axes are [ "Time", "lb", "NuLambda" ].
        """

        if nside_model is None:
            nside_model = full_detector_response.nside

        axis_time = Axis(edges = time_intervals, label = "Time")
        axis_model_map = HealpixAxis(nside = nside_model, coordsys = "galactic", label = "lb")
        axis_local_map = full_detector_response.axes["NuLambda"]

        axis_coordsys_conv_matrix = [ axis_time, axis_model_map, axis_local_map ] #Time, lb, NuLambda

        contents = []

        for i_time, [init_time, end_time] in tqdm(enumerate(axis_time.bounds), total = len(axis_time.bounds)):
            ccm_thispix = np.zeros((axis_model_map.nbins, axis_local_map.nbins)) # without unit

            init_time = Time(init_time, format = 'unix')
            end_time = Time(end_time, format = 'unix')
    
            filtered_orientation = orientation.source_interval(init_time, end_time)

            for ipix in range(hp.nside2npix(nside_model)):
                l, b = hp.pix2ang(nside_model, ipix, nest=is_nest_model, lonlat=True)
                pixel_coord = SkyCoord(l, b, unit = "deg", frame = 'galactic')

                pixel_movement = filtered_orientation.get_target_in_sc_frame(target_name = f"pixel_{ipix}_{i_time}",
                                                                             target_coord = pixel_coord,
                                                                             quiet = True,
                                                                             save = False)

                dwell_time_map = filtered_orientation.get_dwell_map(response = full_detector_response.filename,
                                                                    src_path = pixel_movement,
                                                                    save = False)

                ccm_thispix[ipix] = dwell_time_map.data 
                # (HealpixMap).data returns the numpy array without its unit. dwell_time_map.unit is u.s.

            ccm_thispix_sparse = sparse.COO.from_numpy( ccm_thispix.reshape((1, axis_model_map.nbins, axis_local_map.nbins)) )

            contents.append(ccm_thispix_sparse)

        coordsys_conv_matrix = cls(axis_coordsys_conv_matrix, contents = sparse.concatenate(contents), unit = u.s, sparse = True)
        
        coordsys_conv_matrix.binning_method = "Time"

        return coordsys_conv_matrix

    @classmethod
    def spacecraft_attitude_binning_ccm(cls, full_detector_response, exposure_table, nside_model = None, use_averaged_pointing = False):
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
        use_averaged_pointing : bool, default False
            If it is True, first the averaged Z- and X-pointings are calculated.
            Then the dwell time map is calculated once for ach model pixel and each scatt_binning_index.
            If it is False, the dwell time map is calculated for each attitude in zpointing and xpointing in the exposure table.
            Then the calculated dwell time maps are summed up. 
            In the former case, the computation is fast but may lose the angular resolution. 
            In the latter case, the conversion matrix is more accurate but it takes a long time to calculate it.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix'
            Its axes are [ "ScAtt", "lb", "NuLambda" ].
        """

        if nside_model is None:
            nside_model = full_detector_response.nside
        is_nest_model = True if exposure_table.scheme == 'nest' else False
        nside_local = full_detector_response.nside
        
        n_scatt_bins = len(exposure_table)

        axis_scatt = Axis(edges = np.arange(n_scatt_bins+1), label = "ScAtt")
        axis_model_map = HealpixAxis(nside = nside_model, coordsys = "galactic", scheme = exposure_table.scheme, label = "lb")
        axis_local_map = full_detector_response.axes["NuLambda"]

        axis_coordsys_conv_matrix = [ axis_scatt, axis_model_map, axis_local_map ] #lb, ScAtt, NuLambda
        
        contents = []

        for i_scatt_bin in tqdm(range(n_scatt_bins)):
            ccm_thispix = np.zeros((axis_model_map.nbins, axis_local_map.nbins)) # without unit

            row = exposure_table.iloc[i_scatt_bin]
        
            scatt_binning_index = row['scatt_binning_index']
            num_pointings = row['num_pointings']
            #healpix_index = row['healpix_index']
            zpointing = row['zpointing']
            xpointing = row['xpointing']
            zpointing_averaged = row['zpointing_averaged']
            xpointing_averaged = row['xpointing_averaged']
            delta_time = row['delta_time']
            exposure = row['exposure']
            
            if use_averaged_pointing:
                z = SkyCoord([zpointing_averaged[0]], [zpointing_averaged[1]], frame="galactic", unit="deg")
                x = SkyCoord([xpointing_averaged[0]], [xpointing_averaged[1]], frame="galactic", unit="deg")
            else:
                z = SkyCoord(zpointing.T[0], zpointing.T[1], frame="galactic", unit="deg")
                x = SkyCoord(xpointing.T[0], xpointing.T[1], frame="galactic", unit="deg")
        
            attitude = Attitude.from_axes(x = x, z = z, frame = 'galactic')

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
                    weights = weights * exposure
                else:
                    weights = weights * delta_time

                hist, bins = np.histogram(pixels, bins = axis_local_map.edges, weights = weights)
                
                ccm_thispix[ipix] = hist

            ccm_thispix_sparse = sparse.COO.from_numpy( ccm_thispix.reshape((1, axis_model_map.nbins, axis_local_map.nbins)) )

            contents.append(ccm_thispix_sparse)

        coordsys_conv_matrix = cls(axis_coordsys_conv_matrix, contents = sparse.concatenate(contents), unit = u.s, sparse = True)

        coordsys_conv_matrix.binning_method = 'ScAtt'
        
        return coordsys_conv_matrix
    
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

        new = cls(new.axes, contents = new.contents, sumw2 = new.contents, unit = new.unit) 

        new.binning_method = new.axes.labels[0] # 'Time' or 'ScAtt'

        return new

# TODO
#   def calc_exposure_map(self, full_detector_response): #once the response file format is fixed, I will implement this function
