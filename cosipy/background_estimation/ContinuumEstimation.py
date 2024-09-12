# Imports:
import os
from astropy.coordinates import SkyCoord
from astropy import units as u
from cosipy.response import FullDetectorResponse, DetectorResponse
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import BinnedData
from mhealpy import HealpixMap, HealpixBase
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import numpy.ma as ma
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class ContinuumEstimation:
    
    def calc_psr(self, ori_file, detector_response, coord, output_file, nside=16):

        """Calculates point source response (PSR) in Galactic coordinates.
        
        Parameters
        ----------
        ori_file : str
            Full path to orienation file.
        detector_response : str
            Full path to detector response file.
        coord : tuple
            tuple giving Galactic longitude and latitude of source in degrees: (l,b). 
        nside : int, optional
            nside of scatt map (default is 16). 
        output_file : str
            Prefix of output file (will have .h5 extension).
        """

        # Orientatin file:
        sc_orientation = SpacecraftFile.parse_from_file(ori_file)

        # Detector response:
        dr = detector_response

        # Scatt map:
        scatt_map = sc_orientation.get_scatt_map(nside = nside, coordsys = 'galactic')

        # Calculate PSR:
        coord = coord*u.deg
        coord = SkyCoord(l=coord[0],b=coord[1],frame='galactic')
        with FullDetectorResponse.open(dr) as response:
            self.psr = response.get_point_source_response(coord = coord, scatt_map = scatt_map)

        # Save:
        self.psr.write(output_file + ".h5")

        return 

    def load_psr_from_file(self, psr_file):

        """Loads point source response from h5 file.
            
        Parameters
        ----------
        psr_file : str
            Full path to precomputed response file (.h5 file).
        """

        logger.info("...loading the pre-computed image response ...")
        self.psr = DetectorResponse.open(psr_file)
        logger.info("--> done")

        return

    def load_full_data(self, data_file, data_yaml):

        """Loads binned data to be used as a template for the background estimate.
        
        Parameters
        ----------
        data_file : str
            Full path to binned data (must be .h5 file). 
        data_yaml : str
            Full path to the dataIO yaml file used for binning the data. 

        Notes
        -----
        In practice, the data file used for estimating the background 
        should be the full dataset.  
        
        The full data binning needs to match the PSR. 
        """
        
        self.full_data = BinnedData(data_yaml)
        self.full_data.load_binned_data_from_hdf5(data_file)
        self.estimated_bg = self.full_data.binned_data.project('Em', 'Phi', 'PsiChi')

        return

    def mask_from_cumdist(self, psichi_map, containment, make_plots=False):

        """
        Determines masked pixels from cumulative distribution of
        the point source response.
        
        Parameters
        ----------
        psichi_map : histpy:Histogram
            Point source response projected onto psichi. This can be 
            either a slice of Em and Phi, or the full projection. Note
            that psichi is a HealpixMap axis in histpy. 
        containment : float
            The percentage (non-inclusive) of the cumulative distribution 
            to use for the mask, i.e. all pixels that fall below this value 
            in the cumulative distribution will be masked. 
        make_plots : bool
            Option to plot cumulative distribution. 

        Note
        ----
        The cumulative distribution is an estimate of the angular
        resolution measure (ARM), which is a measure of the PSF
        for Compton imaging. 
        """

        # Get healpix map:
        h = psichi_map
        m = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents)

        # Sort data in descending order:
        sorted_data = np.sort(m)[::-1]

        # Calculte the cummulative distribution
        cumdist = np.cumsum(sorted_data) / sum(sorted_data)

        # Get indices of sorted array
        self.sorted_indices = np.argsort(h.contents.value)[::-1]

        # Define mask based on fraction of total exposure (i.e. counts):
        self.arm_mask = cumdist >= containment
        self.arm_mask = ~self.arm_mask

        # Plot cummulative distribution and corresponding masks:
        if make_plots == True:
            plt.plot(cumdist)
            plt.title("Cumulative Distribution")
            plt.xlabel("Pixel")
            plt.ylabel("Fraction of Counts")
            plt.savefig("cumdist.png")
            plt.show()
            plt.close()
        
        return 

    def simple_inpainting(self, m_data):

        """Highly simplistic method for inpainting masked region in CDS.

        This method relies on the input healpix map having a ring 
        ordering. For each masked pixel, it searches to the left (i.e.
        lower pixel numbers) until reaching the first non-zero pixel. 
        It then search to the right (i.e. higher pixel numbers) until
        again finding the first non-zero pixel. The mean of the two 
        values is used for filling in the masked pixel. 
        
        Parameters
        ----------
        m_data : array-like
            HealpixMap object, containing projection of PSR onto psichi.

        Returns
        -------
        interp_list : array
            Values for the inpainting, corresponding to the masked pixels. 
        """

        # Get mean of masked data for edge cases (simple solution for now):
        # CK: It would be better if this were at least the mean of an 
        # np masked array object, but a better method is anyways needed.
        masked_mean = np.mean(m_data)

        # Get interpolation values:
        interp_list_low = []
        interp_list_high = []
        for i in range(0,len(self.sorted_indices[self.arm_mask])):
            
            this_index = self.sorted_indices[self.arm_mask][i]
            
            # Search left:
            k = 1
            search_left = True
            while search_left == True:
                
                if this_index-k < 0:
                    logger.info("Edge case!")
                    interp_list_low.append(masked_mean)
                    search_left = False
                    break
                    
                next_value = m_data[this_index-k]
                if next_value == 0:
                    k += 1
                if next_value != 0:
                    interp_list_low.append(next_value)
                    search_left = False
           
            # Search right:
            j = 1
            search_right = True
            while search_right == True:
               
                if this_index+j >= len(self.psr.axes['PsiChi'].centers)-1:
                    logger.info("Edge case!")
                    interp_list_high.append(masked_mean)
                    search_right = False
                    break
                
                next_value = m_data[this_index+j]
                if next_value == 0:
                    j += 1
                if next_value != 0:
                    interp_list_high.append(next_value)
                    search_right = False
            
        interp_list_low = np.array(interp_list_low)
        interp_list_high = np.array(interp_list_high)
        interp_list = (interp_list_low + interp_list_high) / 2.0 
    
        return interp_list

    def continuum_bg_estimation(self, data_file, data_yaml, psr_file, \
            output_file, containment=0.4, make_plots=False,\
            e_loop="default", s_loop="default"):

        """Estimates continuum background.
        
        Parameters
        ----------
        data_file : str
            Full path to binned data (must be .h5 file). 
        data_yaml : str
            Full path to the dataIO yaml file used for binning the data.  
        psr_file : str
            Full path to point source respone file. 
        output_file : str
            Prefix of output file for estimated background (will be 
            saved as .h5 file). 
        containment : float, optional
            The percentage (non-inclusive) of the cumulative distribution 
            to use for the mask, i.e. all pixels that fall below this value 
            in the cumulative distribution will be masked. Default is 0.4. 
        make_plots : bool, optional
            Option to make some plots of the data, response, and masks. 
            Default is False.
        e_loop : tuple, optional
            Option to pass tuple specifying which energy range to 
            loop over. This must coincide with the energy bins. The default
            is all bins.
        s_loop : tuple, optional
            Option to pass tuple specifying which Phi anlge range to
            loop over. This must coincide with the Phi  bins. The default
            is all bins.
        """

        # Load data to be used for BG estimation:
        self.laod_full_data(data_file,data_yaml)

        # Load point source respone:
        self.load_psr_from_file(psr_file)

        # Defaults for energy and scattering angle loops:
        if e_loop == "default":
            e_loop = (0,len(self.psr.axes['Em'].centers))
        if s_loop == "default":
            s_loop = (0,len(self.psr.axes['Phi'].centers))

        # Progress bar:
        e_tot = e_loop[1] - e_loop[0]
        s_tot = s_loop[1] - s_loop[0]
        num_lines = e_tot*s_tot
        pbar = tqdm(total=num_lines)

        # Loop through all bins of energy and phi:
        for E in range(e_loop[0],e_loop[1]):
            for s in range(s_loop[0],s_loop[1]):
                
                pbar.update(1) # update progress bar
                logger.info("Bin %s %s" %(str(E),str(s)))

                # Get PSR slice:
                h = self.psr.slice[{'Em':E, 'Phi':s}].project('PsiChi')

                # Get mask:
                self.mask_from_cumdist(h, containment, make_plots=make_plots)       

                # Mask data:
                h_data = self.full_data.binned_data.project('Em', 'Phi', 'PsiChi').slice[{'Em':E, 'Phi':s}].project('PsiChi')
                m_data = HealpixMap(base = HealpixBase(npix = h_data.nbins), data = h_data.contents.todense())
                m_data[self.sorted_indices[self.arm_mask]] = 0

                # Skip this iteration if map is all zeros:
                if len(m_data[m_data[:] > 0]) == 0:
                    logger.info("All zeros and so skipping iteration!")
                    continue

                # Get interpolated values:
                interp_list = self.simple_inpainting(m_data)

                # Update estimated BG:
                for p in range(len(self.sorted_indices[self.arm_mask])):
                    self.estimated_bg[E,s,self.sorted_indices[self.arm_mask][p]] = interp_list[p]

                # Option to make some plots:
                if make_plots == True:
                    
                    # Plot true response:
                    m_dummy = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents)
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("True Response")
                    plt.show()
                    plt.close()

                    # Plot masked response:
                    m_dummy[self.sorted_indices[self.arm_mask]] = 0
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("Masked Response")
                    plt.show()
                    plt.close()

                    # Plot true data:
                    m_data_dummy = HealpixMap(base = HealpixBase(npix = h_data.nbins), data = h_data.contents.todense())
                    plot,ax = m_data_dummy.plot('mollview')
                    plt.title("True Data")
                    plt.show()
                    plt.close()

                    # Plot masked data:
                    plot,ax = m_data.plot('mollview')
                    plt.title("Masked Data")
                    plt.show()
                    plt.close()

                    # Plot masked data with interpolated values:
                    m_data[self.sorted_indices[self.arm_mask]] = interp_list
                    plot,ax = m_data.plot('mollview')
                    plt.title("Interpolated Data (Estimated BG)")
                    plt.show()
                    plt.close()
      
        # Close progress bar:
        pbar.close()

        # Write estimated BG file:
        logger.info("Writing file...")
        self.estimated_bg.write(output_file,overwrite=True)
        logger.info("Finished!")

        return
