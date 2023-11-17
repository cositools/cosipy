import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, cartesian_to_spherical, Galactic
import numpy as np
import healpy as hp
from tqdm.autonotebook import tqdm

from scoords import Attitude, SpacecraftFrame
from histpy import Histogram, Axes, Axis, HealpixAxis

class CoordsysConversionMatrix(Histogram):

    def __init__(self, edges, contents = None, sumw2 = None,
                 labels=None, axis_scale = None, sparse = None, unit = None,
                 binning_method = None):
        
        super().__init__(edges, contents = contents, sumw2 = sumw2,
                         labels = labels, axis_scale = axis_scale, sparse = sparse, unit = unit)

        self.binning_method = None #'Time' or 'Scat'

    @classmethod
    def time_binning_ccm(cls, full_detector_response, orientation, time_intervals, nside_model = None, is_nest_model = False):
        """
        Parameters
        ----------
        full_detector_response: 
        orientation:
        time_intervals: 2d np.array. it is the same format of binned_data.axes['Time'].edges
        nside_model: If it is None, it will be the same as the NSIDE in the response.

        Returns
        -------
        coordsys_conv_matrix: Axes [ "lb", "Time", "NuLambda" ]
        """

        if nside_model is None:
            nside_model = full_detector_response.nside

        axis_model_map = HealpixAxis(nside = nside_model,
                                     coordsys = "galactic", label = "lb")
        axis_time_binning_index = Axis(edges = time_intervals, label = "Time")
        axis_coordsys_conv_matrix = [ axis_model_map, axis_time_binning_index, full_detector_response.axes["NuLambda"] ] #lb, Time, NuLambda

        coordsys_conv_matrix = cls(axis_coordsys_conv_matrix, unit = u.s, sparse = False)

        coordsys_conv_matrix.binning_method = "Time"

        # calculate a dwell time map at each time bin and sky location

        for ipix in tqdm(range(hp.nside2npix(nside_model))):
            l, b = hp.pix2ang(nside_model, ipix, nest=is_nest_model, lonlat=True)
            pixel_coord = SkyCoord(l, b, unit = "deg", frame = 'galactic')

            for i_time, [init_time, end_time] in enumerate(coordsys_conv_matrix.axes["Time"].bounds):
                init_time = Time(init_time, format = 'unix')
                end_time = Time(end_time, format = 'unix')
    
                filtered_orientation = orientation.source_interval(init_time, end_time)
                pixel_movement = filtered_orientation.get_target_in_sc_frame(target_name = f"pixel_{ipix}_{i_time}",
                                                                             target_coord = pixel_coord,
                                                                             quiet = True)

                time_diff = filtered_orientation.get_time_delta()

                dwell_time_map = filtered_orientation.get_dwell_map(response = full_detector_response.filename,
                                                                    dts = time_diff,
                                                                    src_path = pixel_movement,
                                                                    quiet = True)

                coordsys_conv_matrix[ipix,i_time] = dwell_time_map.data * dwell_time_map.unit
                # (HealpixMap).data returns the numpy array without its unit.
        
        coordsys_conv_matrix = coordsys_conv_matrix.to_sparse()

        return coordsys_conv_matrix

    @classmethod
    def scat_binning_ccm(cls, full_detector_response, exposure_table, use_averaged_pointing = False):
        """
        Parameters
        ----------
        full_detector_response: 
        exposure_table:
        use_averaged_pointing: If this is set to True, the ccm loses accuracy but the calculatiion gets much faster.

        Returns
        -------
        coordsys_conv_matrix: Axes [ "lb", "Scat", "NuLambda" ]
        """

        nside_model = exposure_table.nside
        is_nest_model = True if exposure_table.scheme == 'nest' else False
        nside_local = full_detector_response.nside
        
        n_scat_bins = len(exposure_table)

        axis_model_map = HealpixAxis(nside = nside_model, coordsys = "galactic", scheme = exposure_table.scheme, label = "lb")
        axis_scat_binning_index = Axis(edges = np.arange(n_scat_bins+1), label = "Scat")
        axis_coordsys_conv_matrix = [ axis_model_map, axis_scat_binning_index, full_detector_response.axes["NuLambda"] ] #lb, Scat, NuLambda

        coordsys_conv_matrix = cls(axis_coordsys_conv_matrix, unit = u.s, sparse = False)

        coordsys_conv_matrix.binning_method = 'Scat'

        print('... start calculating the coordsys conversion matrix ...')

        for idx in tqdm(range(n_scat_bins)):
            row = exposure_table.iloc[idx]
        
            scat_binning_index = row['scat_binning_index']
            num_pointings = row['num_pointings']
    #           healpix_index = row['healpix_index']
            zpointing = row['zpointing']
            xpointing = row['xpointing']
            delta_time = row['delta_time']
            exposure = row['exposure']
            
            if use_averaged_pointing:
                attitude = coordsys_conv_matrix._get_attitude_using_averaged_pointing(zpointing, xpointing, delta_time)

            else:
                z = SkyCoord(zpointing.T[0], zpointing.T[1], frame="galactic", unit="deg")
                x = SkyCoord(xpointing.T[0], xpointing.T[1], frame="galactic", unit="deg")
        
                attitude = Attitude.from_axes(x = x, z = z, frame = 'galactic')
        
            for ipix in tqdm(range(hp.nside2npix(nside_model))):
                l, b = hp.pix2ang(nside_model, ipix, nest=is_nest_model, lonlat=True)
                pixel_coord = SkyCoord(l, b, unit = "deg", frame = 'galactic')
            
                src_path_cartesian = SkyCoord(np.dot(attitude.rot.inv().as_matrix(), pixel_coord.cartesian.xyz.value),
                                              representation_type = 'cartesian', frame = SpacecraftFrame())
    
                src_path_spherical = cartesian_to_spherical(src_path_cartesian.x, src_path_cartesian.y, src_path_cartesian.z)
    
                l_scr_path = np.array(src_path_spherical[2].deg)  # note that 0 is Quanty, 1 is latitude and 2 is longitude and they are in rad not deg
                b_scr_path = np.array(src_path_spherical[1].deg)
    
                src_path_skycoord = SkyCoord(l_scr_path, b_scr_path, unit = "deg", frame = SpacecraftFrame())
                            
                pixels, weights = coordsys_conv_matrix.axes['NuLambda'].get_interp_weights(src_path_skycoord)
                
                if use_averaged_pointing:
                    weights = weights * exposure * u.s

                else:
                    weights = weights * delta_time * u.s

                hist, bins = np.histogram(pixels, bins = coordsys_conv_matrix.axes['NuLambda'].edges, weights = weights)
                
                coordsys_conv_matrix[ipix,scat_binning_index,:] += hist
            
            print("scat_binning_index:", scat_binning_index)
            print("number of pointings:", num_pointings)
            print("exposure:", np.sum(coordsys_conv_matrix[ipix][scat_binning_index]))
            print("exposure in the table (s):", exposure)
            
        coordsys_conv_matrix = coordsys_conv_matrix.to_sparse()
        
        return coordsys_conv_matrix
    
    def _get_attitude_using_averaged_pointing(self, zpointing, xpointing, delta_time):

        averaged_zpointing = np.sum(hp.ang2vec(zpointing.T[0], zpointing.T[1], lonlat = True).T * delta_time, axis = (1))
        averaged_zpointing /= np.linalg.norm(averaged_zpointing)

        averaged_xpointing = np.sum(hp.ang2vec(xpointing.T[0], xpointing.T[1], lonlat = True).T * delta_time, axis = (1))
        averaged_xpointing /= np.linalg.norm(averaged_xpointing)
        
        averaged_z_l = hp.vec2ang(averaged_zpointing, lonlat = True)[0][0]
        averaged_z_b = hp.vec2ang(averaged_zpointing, lonlat = True)[1][0]
        averaged_x_l = hp.vec2ang(averaged_xpointing, lonlat = True)[0][0]
        averaged_x_b = hp.vec2ang(averaged_xpointing, lonlat = True)[1][0]
    
        zpointing = np.array([[averaged_z_l, averaged_z_b]])
        xpointing = np.array([[averaged_x_l, averaged_x_b]])
        
        z = SkyCoord(zpointing.T[0], zpointing.T[1], frame="galactic", unit="deg")
        x = SkyCoord(xpointing.T[0], xpointing.T[1], frame="galactic", unit="deg")
    
        attitude = Attitude.from_axes(x = x, z = z, frame = 'galactic')

        return attitude
    
    @classmethod
    def open(cls, filename, name = 'hist'):

        new = super().open(filename, name)

        new = cls(new.axes, contents = new.contents, sumw2 = new.contents, unit = new.unit) 

        new.binning_method = new.axes.labels[1]

        return new

#   def calc_exposure_map(self, full_detector_response): #once the response file format is fixed, I will implement this function
