# Imports:
import sys
import numpy as np
import h5py
from histpy import Histogram
from mhealpy import HealpixMap, HealpixBase
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from cosipy.make_plots import MakePlots
from cosipy.data_io import UnBinnedData
import logging
logger = logging.getLogger(__name__)


class BinnedData(UnBinnedData):
   
    def get_binned_data(self, unbinned_data=None, output_name="binned_data", \
            make_binning_plots=False, psichi_binning="galactic"):

        """ 
        Bin the data using histpy and mhealpy.
        
        Optional inputs:
        - unbinned_data: read in unbinned data from file. 
          Input file is either .fits or .hdf5 as specified in
          the unbinned_output parameter in inputs.yaml.     
        - output_name: prefix of output file.
        - make_binning_plots: Option to make basic plots of the binning.
          Default is False.
        - psichi_binning: 'galactic' for binning psichi in Galactic coordinates,
          or 'local' for binning in local coordinates. Default is Galactic. 
        """
        
        # Make print statement:
        print("binning data...")
      
        # Option to read in unbinned data file:
        if unbinned_data:
            if self.unbinned_output == 'fits':
                self.cosi_dataset = self.get_dict_from_fits(unbinned_data)
            if self.unbinned_output == 'hdf5':
                self.cosi_dataset = self.get_dict_from_hdf5(unbinned_data)

        if type(self.time_bins).__name__ in ['int','float']:
            # Get time bins: 
            min_time = self.tmin
            max_time = self.tmax
            delta_t = max_time - min_time
            num_bins = round(delta_t / self.time_bins)
            new_bin_size = delta_t / num_bins
            if self.time_bins != new_bin_size:
                print()
                print("Note: time bins must be equally spaced between min and max time.")
                print("Using time bin size [s]: " + str(new_bin_size))
                print()
            time_bin_edges = np.linspace(min_time,max_time,num_bins+1)

        if type(self.time_bins).__name__ == 'list':
            # Check that bins correspond to min and max time:
            if (self.time_bins[0] > min_time) | (self.time_bins[-1] < max_time):
                print()
                print("ERROR: Time bins do not cover the full selected data range!")
                print()
                sys.exit()
            time_bin_edges = np.array(self.time_bins)

        # Get energy bins:
        energy_bin_edges = np.array(self.energy_bins)

        # Get phi bins:
        number_phi_bins = int(180./self.phi_pix_size)
        phi_bin_edges = np.linspace(0,180,number_phi_bins+1)

        # Get healpix binning for psi and chi local:
        npix = hp.nside2npix(self.nside)
        print()
        print("PsiChi binning:")
        print("Approximate resolution at NSIDE {} is {:.2} deg".format(self.nside, hp.nside2resol(self.nside, arcmin=True) / 60))
        print()

        # Define the grid and initialize
        self.m = HealpixMap(nside = self.nside, scheme = self.scheme, dtype = int)

        # Bin psi and chi data:
        if psichi_binning not in ['galactic','local']:
            print("ERROR: psichi_binning must be either 'galactic' or 'local'")
            sys.exit()
        if psichi_binning == 'galactic':
            PsiChi_pixs = self.m.ang2pix(self.cosi_dataset['Chi galactic'],self.cosi_dataset['Psi galactic'],lonlat=True)
        if psichi_binning == 'local':
            PsiChi_pixs = self.m.ang2pix(self.cosi_dataset['Psi local'],self.cosi_dataset['Chi local'])
        PsiChi_bin_edges = np.arange(0,npix+1,1)
    
        # Fill healpix map:
        unique, unique_counts = np.unique(PsiChi_pixs, return_counts=True)
        self.m[unique] = unique_counts

        # Save healpix map to file:
        self.m.write_map("psichi_healpix_map.fits",overwrite=True)

        # Initialize histogram:
        self.binned_data = Histogram([time_bin_edges, energy_bin_edges, phi_bin_edges, PsiChi_bin_edges], labels = ['Time','Em','Phi','PsiChi'], sparse=True)

        # Fill histogram:
        self.binned_data.fill(self.cosi_dataset['TimeTags'], self.cosi_dataset['Energies'], np.rad2deg(self.cosi_dataset['Phi']), PsiChi_pixs)
 
        # Save binned data to hdf5 file:
        self.binned_data.write('%s.hdf5' %output_name, overwrite=True)

        # Get binning information:
        self.get_binning_info()

        # Plot the binned data:
        if make_binning_plots == True:
            self.plot_binned_data()  
            self.plot_psichi_map()

        return

    def load_binned_data_from_hdf5(self,binned_data):

        """Loads binned histogram from hdf5 file."""
        
        self.binned_data = Histogram.open(binned_data)    

        return

    def get_binning_info(self, binned_data=None):

        """
        Get binning information from Histpy histogram.
        Optional input:
        binned_data: Histogram object (hdf5).
        """
        
        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
        
        # Get time binning information:
        self.time_hist = self.binned_data.project('Time').contents.todense()
        self.num_time_bins = self.binned_data.axes['Time'].nbins
        self.time_bin_centers = self.binned_data.axes['Time'].centers
        self.time_bin_edges = self.binned_data.axes['Time'].edges
        self.time_bin_widths = self.binned_data.axes['Time'].widths
        self.total_time = self.time_bin_edges[-1] - self.time_bin_edges[0]

        # Get energy binning information:
        self.energy_hist = self.binned_data.project('Em').contents.todense()
        self.num_energy_bins = self.binned_data.axes['Em'].nbins
        self.energy_bin_centers = self.binned_data.axes['Em'].centers
        self.energy_bin_edges = self.binned_data.axes['Em'].edges
        self.energy_bin_widths = self.binned_data.axes['Em'].widths

        # Get Phi binning information:
        self.phi_hist = self.binned_data.project('Phi').contents.todense()
        self.num_phi_bins = self.binned_data.axes['Phi'].nbins
        self.phi_bin_centers = self.binned_data.axes['Phi'].centers
        self.phi_bin_edges = self.binned_data.axes['Phi'].edges
        self.phi_bin_widths = self.binned_data.axes['Phi'].widths

        # Get PsiChi binning information:
        self.psichi_hist = self.binned_data.project('PsiChi').contents.todense()
        self.num_psichi_bins = self.binned_data.axes['PsiChi'].nbins
        self.psichi_bin_centers = self.binned_data.axes['PsiChi'].centers
        self.psichi_bin_edges = self.binned_data.axes['PsiChi'].edges
        self.psichi_bin_widths = self.binned_data.axes['PsiChi'].widths
        
        return

    def plot_binned_data(self, binned_data=None):

        """
        Plot binnned data for all axes.
        Optional input:
        binned_data: Histogram object (hdf5).
        """
        
        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
    
        # Define plot dictionaries:
        time_energy_plot = {"projection":["Time","Em"],"xlabel":"Time [s]",\
                "ylabel":"Em [keV]","savefig":"2d_time_energy.png"}
        time_plot = {"projection":"Time","xlabel":"Time [s]",\
                "ylabel":"Counts","savefig":"time_binning.pdf"}
        energy_plot = {"projection":"Em","xlabel":"Em [keV]",\
                "ylabel":"Counts","savefig":"energy_binning.pdf"}
        phi_plot = {"projection":"Phi","xlabel":"Phi [deg]",\
                "ylabel":"Counts","savefig":"phi_binning.pdf"}
        psichi_plot = {"projection":"PsiChi",\
                "xlabel":"PsiChi [HEALPix ring pixel number]",\
                "ylabel":"Counts","savefig":"psichi_binning.pdf"}
        
        # Make plots:
        plot_list = [time_energy_plot,time_plot,energy_plot,phi_plot,psichi_plot]
        for each in plot_list:
            self.binned_data.project(each["projection"]).plot()
            plt.xlabel(each["xlabel"],fontsize=12)
            plt.ylabel(each["ylabel"], fontsize=12)
            plt.savefig(each["savefig"])
            plt.show()
            plt.close() 
 
        return

    def plot_psichi_map(self, healpix_map=None):

        """
        Plot psichi healpix map.
        Optional input:
        healpix_map: Healpix map object.
        """

        # Option to read in healpix map from fits file.
        if healpix_map:
            self.m = HealpixMap(nside = self.nside, scheme = self.scheme, dtype = int).read_map(healpix_map)

        # Plot PsiChi mhealpy default:
        plot,ax = self.m.plot()
        ax.get_figure().set_figwidth(4) 
        ax.get_figure().set_figheight(3)
        plt.title("PsiChi Binning (counts)")
        plt.savefig("psichi_default.png",bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot PsiChi mhealpy rotated:
        plot,ax = self.m.plot(ax = 'orthview', ax_kw = {'rot':[0,90,0]})
        ax.get_figure().set_figwidth(3)  
        ax.get_figure().set_figheight(4)
        plt.title("PsiChi Binning (counts)")
        plt.savefig("psichi_rotated.png",bbox_inches='tight')
        plt.show()
        plt.close()
    
        return

    def plot_psichi_map_slices(self, Em, phi, output, binned_data=None, coords=None):

        """
        Plot psichi map in slices of Em and phi.
        Inputs:
        Em: Bin of energy slice.
        phi: Bin of phi slice.
        output: Name of output plot. 
        binned_data (optional): Histogram object (hdf5).
        coords (optional; list): Coordinates of source position. 
            - Galactic longitude and latidude for Galactic coordinates.
            - Azimuthal and latitude for local coordinates. 
        """

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)

        # Make healpix map with binned data slice:
        h = self.binned_data.project('Em', 'Phi', 'PsiChi').slice[{'Em':Em, 'Phi':phi}].project('PsiChi')
        m = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents.todense())
        
        # Plot standard view:
        plot,ax = m.plot('mollview')
        if coords:
            ax.scatter(coords[0], coords[1], s=9, transform=ax.get_transform('world'), color = 'red')
        ax.coords.grid(True, color='grey', ls='dotted')
        ax.get_figure().set_figwidth(6)
        ax.get_figure().set_figheight(3)
        plt.savefig("%s.pdf" %output,bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot rotated view:
        if coords:
            plot,ax = m.plot('orthview', ax_kw = {'rot':[coords[0],coords[1],0]})
            ax.scatter(coords[0], coords[1], s=9, transform=ax.get_transform('world'), color = 'red')
            ax.coords.grid(True, color='grey', ls='dotted')
            ax.get_figure().set_figwidth(6)
            ax.get_figure().set_figheight(3)
            plt.savefig("%s_rotated.pdf" %output,bbox_inches='tight')
            plt.show()
            plt.close()

        return

    def get_raw_spectrum(self, binned_data=None, time_rate=False, output_name="raw_spectrum"):

        """
        Calculates raw spectrum of binned data, plots, and writes to file. 
        
        Inputs (all optional):
        binned_data: Binnned data file (hdf5).
        output_name: Prefix of output files. 
        time_rate: If True calculates ct/keV/s. The defualt is ct/keV. 
        """

        # Make print statement:
        print("getting raw spectrum...")

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
            self.get_binning_info() 

        # Option to normalize by total time:
        if time_rate==False:
            raw_rate = self.energy_hist/self.energy_bin_widths
            ylabel = "$\mathrm{ct \ keV^{-1}}$"
            data_label = "Rate[ct/keV]"
        if time_rate==True:
            raw_rate = self.energy_hist/self.energy_bin_widths/self.total_time
            ylabel = "$\mathrm{ct \ keV^{-1} \ s^{-1}}$"
            data_label = "Rate[ct/keV/s]"

        # Plot:
        plot_kwargs = {"label":"raw spectrum", "ls":"", "marker":"o", "color":"black"}
        fig_kwargs = {"xlabel":"Energy [keV]", "ylabel":ylabel}
        MakePlots().make_basic_plot(self.energy_bin_centers, raw_rate,\
            x_error=self.energy_bin_widths/2.0, savefig="%s.pdf" %output_name,\
            plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs)

        # Write data:
        d = {"Energy[keV]":self.energy_bin_centers,data_label:raw_rate}
        df = pd.DataFrame(data=d)
        df.to_csv("%s.dat" %output_name,float_format='%10.5e',index=False,sep="\t",columns=["Energy[keV]",data_label])
        
        return

    def get_raw_lightcurve(self, binned_data=None, output_name="raw_lc"):

        """
        Calculates raw lightcurve of binned data, plots, and writes data to file.
        
        Inputs (optional):
        binned_data: Binnned data file (hdf5).
        """

        # Make print statement:
        print("getting raw lightcurve...")

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
            self.get_binning_info() 
        
        # Calculate raw light curve:
        raw_lc = self.time_hist/self.time_bin_widths

        # Plot:
        plot_kwargs = {"ls":"-", "marker":"", "color":"black", "label":"raw lightcurve"}
        fig_kwargs = {"xlabel":"Time [s]", "ylabel":"Rate [$\mathrm{ct \ s^{-1}}$]"}
        MakePlots().make_basic_plot(self.time_bin_centers - self.time_bin_centers[0], raw_lc,\
            savefig="%s.pdf" %output_name, plt_scale="semilogy", plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs)
            
        # Write data:
        d = {"Time[UTC]":self.time_bin_centers,"Rate[ct/s]":self.time_hist/self.time_bin_widths}
        df = pd.DataFrame(data=d)
        df.to_csv("%s.dat" %output_name,index=False,sep="\t",columns=["Time[UTC]","Rate[ct/s]"])

        return
