# Imports:
import sys
import numpy as np
import h5py
from histpy import Histogram, HealpixAxis, Axis
from scoords import SpacecraftFrame, Attitude
from mhealpy import HealpixMap, HealpixBase
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from cosipy.data_io import UnBinnedData
import logging
import astropy.units as u
from astropy.coordinates import SkyCoord
logger = logging.getLogger(__name__)


class BinnedData(UnBinnedData):
    """Handles binned data."""

    def get_binned_data(self, unbinned_data=None, output_name=None, \
            make_binning_plots=False, show_plots=False, \
            psichi_binning="galactic", event_range=None):

        """Bin the data using histpy and mhealpy.
        
        Parameters
        ----------
        unbinned_data : str, optional
            Name of unbinned data file to use. Input file is either 
            .fits or .hdf5 as specified in the unbinned_output 
            parameter in inputs.yaml.     
        output_name : str, optional
            Prefix of output file.
        make_binning_plots : bool, optional
            Option to make basic plots of the binning (default is False).
        show_plots : bool, optional
            Option to show plots (default is False).
        psichi_binning : str, optional
            'galactic' for binning psichi in Galactic coordinates, or 
            'local' for binning in local coordinates. Default is Galactic. 
        event_range : list of integers, optional 
            min and max event to use for the binning. 
        
        Returns
        -------
        binned_data : histpy:Histogram
            Data is binned in four axes: time, measured energy, 
            Compton scattering angle (phi), and scattering direction 
            (PsiChi).

        Note
        ----
        This method constructs the instance attribute, binned_data, 
        but it does not explicitly return it.
        """
        
        # Log message:
        logger.info("binning data...")

        # Option to read in unbinned data file:
        if unbinned_data:
            self.cosi_dataset = self.get_dict(unbinned_data)

        # Get time bins:
        min_time = self.tmin
        max_time = self.tmax
        if type(self.time_bins).__name__ in ['int','float']:
            # Get time bins: 
            delta_t = max_time - min_time
            num_bins = round(delta_t / self.time_bins)
            new_bin_size = delta_t / num_bins
            if self.time_bins != new_bin_size:
                logger.info("Note: time bins must be equally spaced between min and max time.")
                logger.info("Using time bin size [s]: " + str(new_bin_size))
            time_bin_edges = np.linspace(min_time,max_time,num_bins+1)

        if type(self.time_bins).__name__ == 'list':
            # Check that bins correspond to min and max time:
            if (self.time_bins[0] > min_time) | (self.time_bins[-1] < max_time):
                logger.error("ERROR: Time bins do not cover the full selected data range!")
                sys.exit()
            time_bin_edges = np.array(self.time_bins)

        # Get energy bins:
        energy_bin_edges = np.array(self.energy_bins)

        # Get phi bins:
        number_phi_bins = int(180./self.phi_pix_size)
        phi_bin_edges = np.linspace(0,180,number_phi_bins+1)

        # Define psichi axis and data for binning:
        if psichi_binning == 'galactic':
            psichi_axis = HealpixAxis(nside = self.nside, 
                    scheme = self.scheme, coordsys = 'galactic', label='PsiChi')
            coords = SkyCoord(l=self.cosi_dataset['Chi galactic']*u.deg, 
                    b=self.cosi_dataset['Psi galactic']*u.deg, frame = 'galactic')
        if psichi_binning == 'local':
            psichi_axis = HealpixAxis(nside = self.nside, 
                    scheme = self.scheme, coordsys = SpacecraftFrame(), label='PsiChi')
            coords = SkyCoord(lon=self.cosi_dataset['Chi local']*u.rad, 
                    lat=((np.pi/2.0) - self.cosi_dataset['Psi local'])*u.rad, 
                    frame = SpacecraftFrame())

        # Initialize histogram:
        self.binned_data = Histogram([Axis(time_bin_edges*u.s, label='Time'), 
            Axis(energy_bin_edges*u.keV, label='Em'), 
            Axis(phi_bin_edges*u.deg, label='Phi'), 
            psichi_axis], 
            sparse=True)
         
        # Fill histogram:
        if event_range == None:
            self.binned_data.fill(self.cosi_dataset['TimeTags']*u.s, 
                    self.cosi_dataset['Energies']*u.keV, 
                    np.rad2deg(self.cosi_dataset['Phi'])*u.deg, 
                    coords)
        if event_range != None:
            low = int(event_range[0])
            high = int(event_range[1])
            self.binned_data.fill(self.cosi_dataset['TimeTags'][low:high]*u.s, 
                    self.cosi_dataset['Energies'][low:high]*u.keV, 
                    np.rad2deg(self.cosi_dataset['Phi'][low:high])*u.deg, 
                    coords[low:high])

        # Save binned data to hdf5 file:
        if output_name != None:
            self.binned_data.write('%s.hdf5' %output_name, overwrite=True)

        # Get binning information:
        self.get_binning_info()

        # Plot the binned data:
        if make_binning_plots == True:
            self.plot_binned_data(show_plots=show_plots)  

        return

    def load_binned_data_from_hdf5(self,binned_data):

        """Loads binned histogram from hdf5 file.
        
        Parameters
        ----------
        binned_data : str
            Name of binned data file to load. 

        Returns
        -------
        binned_data : histpy:Histogram
            Data is binned in four axes: time, measured energy, 
            Compton scattering angle (phi), and scattering direction 
            (PsiChi).

        Note
        ----
        This method sets the instance attribute, binned_data, 
        but it does not explicitly return it.
        """
        
        self.binned_data = Histogram.open(binned_data)    

        return

    def get_binning_info(self, binned_data=None):

        """Get binning information from Histpy histogram.
        
        Parameters
        ----------
        binned_data : str
            Name of binned data hdf5 file to use.
        """
        
        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
       
        # Print units of axes:
        for each in self.binned_data.axes:
            logger.info(each.label + " unit: " + str(each.unit))

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

    def plot_binned_data(self, binned_data=None, show_plots=True):

        """Plot binnned data for all axes.
        
        Parameters
        ----------
        binned_data : histpy:Histogram, optional
            Name of binned histogram to use. 
        show_plots : bool, optional
            Option to show plots (default is True).
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
            if each["projection"] == "PsiChi":
                coordsys = self.binned_data.axes['PsiChi'].coordsys
                logger.info(f"plotting psichi map in {coordsys.name}...")
                self.binned_data.project(each["projection"]).plot(coord = 'G', ax_kw = {'coord':'G'})
            else: 
                self.binned_data.project(each["projection"]).plot()
            plt.xlabel(each["xlabel"],fontsize=12)
            plt.ylabel(each["ylabel"], fontsize=12)
            plt.savefig(each["savefig"])
            if show_plots == True:
                plt.show()
            plt.close() 
 
        return

    def plot_psichi_map(self, binned_data=None, show_plots=True):
        
        """
        Plot psichi healpix map.

        Parameters
        ----------
        binned_data : histpy:Histogram, optional
            Name of binned histogram to use. 
        show_plots : bool, optional
            Option to show plot (default is True).
        """

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
    
        # Get psichi coordinate frame: 
        coordsys = self.binned_data.axes['PsiChi'].coordsys
        logger.info(f"plotting psichi map in {coordsys.name}...")
 
        plot, ax = self.binned_data.project('PsiChi').plot(coord = 'G', ax_kw = {'coord':'G'})

        ax.get_figure().set_figwidth(4)
        ax.get_figure().set_figheight(3)
        plt.title("PsiChi Binning (counts)")
        plt.savefig("psichi_default.png",bbox_inches='tight')
        if show_plots == True:
            plt.show()
        plt.close()

        return

    def plot_psichi_map_slices(self, Em, phi, output, binned_data=None, \
            coords=None, show_plots=True):

        """Plot psichi map in slices of Em and phi.
        
        Parameters
        ----------
        Em : int 
            Bin of energy slice.
        phi : int 
            Bin of phi slice.
        output : str
            Prefix of output plot. 
        binned_data : histpy:Histogram, optional
            Name of binned histogram to use. 
        coords : list, optional
            Coordinates of source position. Galactic longitude and 
            latidude for Galactic coordinates. Azimuthal and latitude 
            for local coordinates. 
        show_plots : bool, optional
            Option to show plots (default is True).
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
        if show_plots == True:
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
            if show_plots == True:
                plt.show()
            plt.close()

        return

    def make_basic_plot(self, x, y, plt_scale='loglog', output_name=None,\
        x_error=[], plot_kwargs={}, fig_kwargs={}, show_plots=True):

        """Make a basic plot.

        Parameters
        ----------
        x : list or array
            x-axis data to be plotted
        y : list or array 
            y-axis data to be plotted
        plt_scale : str, optional
            scale of axes: loglog, semilogx, or semilogy.
            Default is loglog.
        x_error : list or array, optional 
            x error bars
        output_name : str, optional
            Prefix of saved figure (default is None).
        plot_kwargs : dict, optional
            Pass any kwargs to plt.plot().
        fig_kwargs : dict, optional 
            Pass any kwargs to plt.gca().set().
        show_plots : bool
            Wether or not to show plot (default is True).
        """
        
        # Setup figure:
        ax = plt.gca()

        # Main plot:
        if plt_scale == "loglog":
            plt.loglog(x, y, **plot_kwargs)
        if plt_scale == "semilogx":
            plt.semilogx(x, y, **plot_kwargs)
        if plt_scale == "semilogy":
            plt.semilogy(x, y, **plot_kwargs)

        # Include x error bars:
        if len(x_error) != 0:
            
            # Remove label if defined:
            if "label" in plot_kwargs.keys():
                plot_kwargs["label"] = "_nolabel_"

            plt.errorbar(x, y, xerr=x_error, **plot_kwargs)

        # axes and labels:
        plt.grid(color="grey",alpha=0.3,ls="--")
        ax.set(**fig_kwargs)
        if "label" in plot_kwargs.keys():
            plt.legend(loc=1,frameon=True)
        
        # Save and show:
        if output_name != None:
            plt.savefig("%s.pdf" %output_name)
        if show_plots == True:
            plt.show()
        plt.close()

        return
    
    def get_raw_spectrum(self, binned_data=None, time_rate=False, output_name=None, show_plots=False):

        """Calculates raw spectrum of binned data, plots, and writes to file. 
        
        Parameters
        ----------
        binned_data : str, optional
            Name of binnned hdf5 data file.
        output_name : str, optional
            Prefix of output files. Writes both pdf and dat file. 
        time_rate : bool, optional
            If True, calculates ct/keV/s. The defualt is ct/keV. 
        show_plot : bool, optional
            Wether or not to show plot (default is False).
        """

        # Log message:
        logger.info("getting raw spectrum...")

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
        self.make_basic_plot(self.energy_bin_centers, raw_rate,\
            x_error=self.energy_bin_widths/2.0, output_name=output_name,\
            plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs, show_plots=show_plots)

        # Write data:
        if output_name != None:
            d = {"Energy[keV]":self.energy_bin_centers,data_label:raw_rate}
            df = pd.DataFrame(data=d)
            df.to_csv("%s.dat" %output_name,float_format='%10.5e',index=False,sep="\t",columns=["Energy[keV]",data_label])
        
        return

    def get_raw_lightcurve(self, binned_data=None, output_name=None, show_plots=False):

        """Calculates raw lightcurve of binned data, plots, and writes data to file.
        
        Parameters
        ----------
        binned_data : str, optional
            Name of binnned hdf5 data file to use.
        output_name : str, optional
            Prefix of output files. Writes both pdf and dat file. 
        show_plots : bool, optional
            Wether or not to show plot (default is False).
        """

        # Log message:
        logger.info("getting raw lightcurve...")

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.load_binned_data_from_hdf5(binned_data)
            self.get_binning_info() 
        
        # Calculate raw light curve:
        raw_lc = self.time_hist/self.time_bin_widths

        # Plot:
        plot_kwargs = {"ls":"-", "marker":"", "color":"black", "label":"raw lightcurve"}
        fig_kwargs = {"xlabel":"Time [s]", "ylabel":"Rate [$\mathrm{ct \ s^{-1}}$]"}
        self.make_basic_plot(self.time_bin_centers - self.time_bin_centers[0], raw_lc,\
            output_name=output_name, plt_scale="semilogy", 
            plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs, show_plots=show_plots)
            
        # Write data:
        if output_name != None:
            d = {"Time[UTC]":self.time_bin_centers,"Rate[ct/s]":self.time_hist/self.time_bin_widths}
            df = pd.DataFrame(data=d)
            df.to_csv("%s.dat" %output_name,index=False,sep="\t",columns=["Time[UTC]","Rate[ct/s]"])

        return
