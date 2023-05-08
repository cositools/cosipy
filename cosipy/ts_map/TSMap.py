from cosipy.threeml.COSILike import COSILike

from threeML import DataList, Powerlaw, PointSource, Model, JointLikelihood

import numpy as np

from histpy import Histogram, Axis

from scipy import stats

import matplotlib.pyplot as plt

import astropy.io.fits as fits


class TSMap:
    
    def __init__(self, *args, **kwargs):
        pass
    
    def link_model_all_plugins(self, dr, data, bkg, sc_orientation, piv, index, other_plugins=None, norm=1, ra=0, dec=0):
        
        # necessary inputs
        self.dr             = dr
        self.data           = data
        self.bkg            = bkg
        self.sc_orientation = sc_orientation
        self.piv            = piv
        self.index          = index
        
        # optional inputs (have default value)
        self.other_plugins  = other_plugins
        self.norm           = norm
        self.ra             = ra
        self.dec            = dec
        
        # instantiate plugin by dr, data, bkg and sc_orientation
        # only COSI plugin for now
        self.instantiate_plugin()
        
        # gather all plugins 
        # only COSI plugin for now
        self.gather_all_plugins()
        
        # create model by Powerlaw and PointSource 
        # Powerlaw needs norm (free parameter), piv and index (free parameter)
        # PointSource needs ra and dec
        self.create_model()
        
        # fix index in further 3ML fitting
        self.fix_index()
        
        # put model and all plugins together
        self.like = JointLikelihood(self.model, self.all_plugins, verbose = False)
        
    def instantiate_plugin(self):
        
        if self.other_plugins == None:
            self.cosi_plugin = COSILike("cosi",
                                        dr = self.dr,
                                        data = self.data, 
                                        bkg = self.bkg, 
                                        sc_orientation = self.sc_orientation)
        else:
            raise RuntimeError("Only COSI plugin for now")
            
    def gather_all_plugins(self):
        
        if self.other_plugins == None:
            self.all_plugins = DataList(self.cosi_plugin)
        else:
            raise RuntimeError("Only COSI plugin for now")
    
    def create_model(self):
        
        self.spectrum = Powerlaw()
        
        self.spectrum.K.value = self.norm # 1/keV/cm2/s
        self.spectrum.piv.value = self.piv # keV
        self.spectrum.index.value = self.index
        
        self.source = PointSource("source", # The name of the source is arbitrary, but needs to be unique
                                  ra = self.ra, 
                                  dec = self.dec,
                                  spectral_shape = self.spectrum)
        
        self.model = Model(self.source)
        
    def fix_index(self):
        
        self.source.spectrum.main.Powerlaw.index.fix = True
    
    def ts_fitting(self):
        
        # collect ts_grid_data, ts_grid_bkg and calculate_ts because sometime we may want to skip fiiting
        self.ts_grid_data()
        self.ts_grid_bkg()
        self.calculate_ts()
        
    # iterate ra and dec to find the best fit of data (time consuming)
    def ts_grid_data(self):
        
        # using rad due to mollweide projection
        self.ra_range  = (-np.pi  , np.pi  ) # rad
        self.dec_range = (-np.pi/2, np.pi/2) # rad
        
        self.log_like = Histogram(
            [Axis(np.linspace(*self.ra_range , 50), label = "ra" ), 
             Axis(np.linspace(*self.dec_range, 25), label = "dec"),]
        )
        
        for i in range(self.log_like.axes['ra'].nbins):
            for j in range(self.log_like.axes['dec'].nbins):
        
                # progress
                print(f"\rra = {i:2d}/{self.log_like.axes['ra'].nbins}   ", end = "")
                print(f"dec = {j:2d}/{self.log_like.axes['dec'].nbins}   ", end = "")
        
                # changing the position parameters
                # converting rad to deg due to ra and dec in 3ML PointSource
                if self.log_like.axes['ra'].centers[i] < 0:
                    self.source.position.ra = (self.log_like.axes['ra'].centers[i] + 2*np.pi) * (180/np.pi) # deg
                else:
                    self.source.position.ra = (self.log_like.axes['ra'].centers[i]) * (180/np.pi) # deg
                self.source.position.dec = self.log_like.axes['dec'].centers[j] * (180/np.pi) # deg
                
                # maximum likelihood
                self.like.fit(quiet=True)
                
                # converting the min (- log likelihood) from 3ML to the max log likelihood for TS 
                self.log_like[i, j] = -self.like._current_minimum 
        
    # iterate ra and dec to find the best fit of bkg
    # only see it as constant for now
    # set the normalization to 0, that is, background-only null-hypothesis
    def ts_grid_bkg(self):
        
        # spectrum.K.value need to be 1e-10 otherwise you will have a migrad error
        self.spectrum.K.value = 1e-10
        
        # maximum likelihood
        self.like.fit(quiet=True)
        
        # converting the min (- log likelihood) from 3ML to the max log likelihood for TS 
        self.log_like0 = -self.like._current_minimum 
        
    # calculate TS by ts_grid_data and ts_grid_bkg
    def calculate_ts(self):
        
        self.ts = 2 * (self.log_like - self.log_like0)

        # getting the maximum
        # note that, in our case, since log_like0 is a constant, max(TS) = 2
        self.argmax = np.unravel_index(np.argmax(self.ts), self.ts.nbins)
        self.ts_max = self.ts[self.argmax]
        
    def print_best_fit(self):
        
        # report the best fit position
        # converting rad to deg due to ra and dec in 3ML PointSource
        if self.ts.axes['ra'].centers[self.argmax[0]] < 0:
            self.best_ra = (self.ts.axes['ra'].centers[self.argmax[0]] + 2*np.pi) * (180/np.pi) # deg
        else:
            self.best_ra = (self.ts.axes['ra'].centers[self.argmax[0]]) * (180/np.pi) # deg
        self.best_dec = self.ts.axes['dec'].centers[self.argmax[1]] * (180/np.pi) # deg
        print(f"Best fit position: RA = {self.best_ra} deg, Dec = {self.best_dec} deg")
        
        # convert to significance based on Wilk's theorem
        print(f"Expected significance: {stats.norm.isf(stats.chi2.sf(self.ts_max, df = 2)):.1f} sigma")
        
    def save_ts(self, output_file_name):
        
        # save TS to .h5 file
        self.ts.write(output_file_name, overwrite = True)
    
    def load_ts(self, input_file_name):
        
        # load .h5 file to TS
        self.ts = Histogram.open(input_file_name)
        
        # getting the maximum
        self.argmax = np.unravel_index(np.argmax(self.ts), self.ts.nbins)
        self.ts_max = self.ts[self.argmax]
    
    # refit the best fit to check norm
    def refit_best_fit(self):
        
        # reset self.spectrum.K.value to self.norm (big initial value)
        self.spectrum.K.value = self.norm
        
        # converting rad to deg due to RA and Dec in 3ML PointSource
        if self.ts.axes['ra'].centers[self.argmax[0]] < 0:
            self.source.position.ra = (self.ts.axes['ra'].centers[self.argmax[0]] + 2*np.pi) * (180/np.pi) # deg
        else:
            self.source.position.ra = (self.ts.axes['ra'].centers[self.argmax[0]]) * (180/np.pi) # deg
        self.source.position.dec = self.ts.axes['dec'].centers[self.argmax[1]] * (180/np.pi) # deg
    
        # maximum likelihood
        self.like.fit()
    
        # display the best fit result 
        self.like.results.display()

    def plot_ts_map(self):
        
        fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={'projection': 'mollweide'}, dpi=120)
        
        _,plot = self.ts.plot(ax, vmin = 0, colorbar = False, zorder=0)
        
        ax.scatter([self.ts.axes['ra'].centers[self.argmax[0]]],[self.ts.axes['dec'].centers[self.argmax[1]]], label = "Max TS", zorder=3)
        
        ax.scatter([20/180*np.pi],[40/180*np.pi], marker = "x", label = "Injected", zorder=2)
        
        # here we also use Wilk's theorem to find the DeltaTS that corresponse to a 90% containment confidence
        ts_thresh = self.ts_max - stats.chi2.isf(1-.9, df = 2)
        contours = ax.contour(self.ts.axes['ra'].centers, 
                              self.ts.axes['dec'].centers, 
                              self.ts.contents.transpose(), 
                              [ts_thresh], colors = 'red', zorder=1)
        contours.collections[0].set_label("90% cont.")
        
        cbar = fig.colorbar(plot)
        cbar.ax.set_ylabel("TS")
        
        ax.set_xlabel('R.A.', fontsize=15);
        ax.set_ylabel('Dec.', fontsize=15);
        ax.tick_params(axis='x', colors='White')
        ax.legend(fontsize=10)
        
    