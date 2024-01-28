from .PointSourceResponse import PointSourceResponse
from .DetectorResponse import DetectorResponse
from astromodels.core.model_parser import ModelParser
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import astropy.units as u
from sparse import COO
from pathlib import Path
import numpy as np
import mhealpy as hp
from mhealpy import HealpixBase, HealpixMap
from cosipy.config import Configurator

from scoords import SpacecraftFrame, Attitude

from histpy import Histogram, Axes, Axis, HealpixAxis
import h5py as h5
import os
import textwrap
import argparse
import logging
logger = logging.getLogger(__name__)

from copy import copy, deepcopy
import gzip
from tqdm import tqdm
import subprocess
import sys
import pathlib
import gc

class FullDetectorResponse(HealpixBase):
    """
    Handles the multi-dimensional matrix that describes the
    full all-sky response of the instrument.

    You can access the :py:class:`DetectorResponse` at a given pixel using the ``[]``
    operator. Alternatively you can obtain the interpolated reponse using
    :py:func:`get_interp_response`.
    """

    def __init__(self, *args, **kwargs):
        # Overload parent init. Called in class methods.
        pass

    @classmethod
    def open(cls, filename,Spectrumfile=None,norm="Linear" ,single_pixel = False,alpha=0,emin=90,emax=10000):
        """
        Open a detector response file.

        Parameters
        ----------
        filename : str, :py:class:`~pathlib.Path`
        Path to the response file (.h5 or .rsp)

        Spectrumfile : str, 
             path to the input spectrum file used
             for the simulation (optional).

         norm : str, 
             type of normalisation : file (then specify also SpectrumFile)
             ,powerlaw, Mono or Linear
         
         alpha : int,
             if the normalisation is "powerlaw", value of the spectral index.

         single_pixel : bool,
             True if there is only one pixel and not full-sky.

         emin,emax : float
             emin/emax used in the simulation source file.  
        
        """
        
        filename = Path(filename)


        if filename.suffix == ".h5":
            return cls._open_h5(filename)
        elif "".join(filename.suffixes[-2:]) == ".rsp.gz":
            return cls._open_rsp(filename,Spectrumfile,norm ,single_pixel,alpha,emin,emax)
        else:
            raise ValueError(
                "Unsupported file format. Only .h5 and .rsp.gz extensions are supported.")

    @classmethod
    def _open_h5(cls, filename):
        """
         Open a detector response h5 file.

         Parameters
         ----------
         filename : str, :py:class:`~pathlib.Path`
             Path to HDF5 file
         """
        new = cls(filename)

        new._file = h5.File(filename, mode='r')

        new._drm = new._file['DRM']

        new._unit = u.Unit(new._drm.attrs['UNIT'])
        
        try:
             new._sparse = new._drm.attrs['SPARSE']
        except KeyError:
             new._sparse = True

        # Axes
        axes = []

        for axis_label in new._drm["AXES"]:

            axis = new._drm['AXES'][axis_label]

            axis_type = axis.attrs['TYPE']

            if axis_type == 'healpix':

                axes += [HealpixAxis(edges=np.array(axis),
                                         nside=axis.attrs['NSIDE'],
                                         label=axis_label,
                                         scheme=axis.attrs['SCHEME'],
                                         coordsys=SpacecraftFrame())]

            else:
                axes += [Axis(np.array(axis) * u.Unit(axis.attrs['UNIT']),
                                  scale=axis_type,
                                  label=axis_label)]

        new._axes = Axes(axes)

        # Init HealpixMap (local coordinates, main axis)
        HealpixBase.__init__(new,
                                 base=new.axes['NuLambda'],
                                 coordsys=SpacecraftFrame())

        return new

    @classmethod
    def _open_rsp(cls, filename, Spectrumfile=None,norm="Linear" ,single_pixel = False,alpha=0,emin=90,emax=10000):
        """
        
         Open a detector response rsp file.

         Parameters
         ----------
         filename : str, :py:class:`~pathlib.Path`
             Path to rsp file

         Spectrumfile : str, 
             path to the input spectrum file used
             for the simulation (optional).

         norm : str, 
             type of normalisation : file (then specify also SpectrumFile)
             ,powerlaw, Mono or Linear
         
         alpha : int,
             if the normalisation is "powerlaw", value of the spectral index.

         single_pixel : bool,
             True if there is only one pixel and not full-sky.

         emin,emax : float
             emin/emax used in the simulation source file.
         
        """
        labels = ("Ei", "NuLambda", "Em", "Phi", "PsiChi", "SigmaTau", "Dist")

        axes_edges = []
        axes_types = []
        sparse = None
        # get the header infos of the rsp file (nsim,area,bin_edges,etc...)
        with gzip.open(filename, "rt") as file:
            for n, line in enumerate(file):
    
                line = line.split()

                if len(line) == 0:
                    continue

                key = line[0]

                if key == 'TS':
                    nevents_sim = int(line[1])

                elif key == 'SA':
                    area_sim = float(line[1])
                    
                elif key == "SP" :
                    
                    try :
                        norm = str(line[1])
                    except :
                        print(f"norm not found in the file ! We assume {norm}")

                    if norm =="Linear" :
                        emin = int(line[2])
                        emax = int(line[3])
                        
             
		
                elif key == "MS":
                    if line[1] == "true" :
                        sparse = True
                    if line[1] == "false" :
                        sparse = False

                elif key == 'AD':

                    if axes_types[-1] == "FISBEL":

                        raise RuntimeError("FISBEL binning not currently supported")
                        
                    elif axes_types[-1] == "HEALPix":

                        if line[2] != "RING":
                            raise RuntimeError(f"Scheme {line[2]} not supported")

                        if line[1] == '-1':
                            # Single bin axis --i.e. all-sky
                            axes_edges.append(-1)
                        else:
                            nside = int(2**int(line[1]))
                            axes_edges.append(int(12*nside**2))
                        
                        
                    else:
                        
                        axes_edges.append(np.array(line[1:], dtype='float'))

                elif key == 'AT':
                    axes_types += [line[2]]

                elif key == 'RD':
                    break
                
                elif key == "StartStream":
                    nbins = int(line[1])
                    break
        
        #check if the type of spectrum is known
        assert norm=="powerlaw" or norm=="Mono" or norm=="Linear","unknown normalisation !" 
         
        #check the number of simulated events is not 0
        assert nevents_sim != 0,"number of simulated events is 0 !" 
        
        
        print("normalisation is {0}".format(norm))
        if sparse == None :
            print("Sparse paramater not found in the file : We assume this is a non sparse matrice !")
            sparse = False
        else :
            print("Sparse matrice ? {0}".format(sparse))
        edges = ()
        #print(axes_edges)

        for axis_edges, axis_type in zip(axes_edges, axes_types):

            if axis_type == 'HEALPix':

                if axis_edges == -1:
                    # Single bin axis --i.e. all-sky
                    edges += ([0,1],)
                else:
                    edges += (np.arange(axis_edges+1),)

            elif axis_type == "FISBEL":
                raise RuntimeError("FISBEL binning not currently supported")
            else:
                edges += (axis_edges,)

        #print(edges)
        
        if sparse :
            axes = Axes(edges, labels=labels)
        
        else :
            axes = Axes(edges[:-2], labels=labels[:-2])            


        if sparse :
            # Need to get number of lines for progress bar.
            # First try fast method for unix-based systems:
            try:
                proc=subprocess.Popen('gunzip -c %s | wc -l' %filename, \
                        shell=True, stdout=subprocess.PIPE)
                nlines = int(proc.communicate()[0])


            # If fast method fails, use long method, which should work in all cases.
            except:
                print("Initial attempt failed.")
                print("Using long method...")
                nlines = sum(1 for _ in gzip.open(filename,"rt"))
                
            # Preallocate arrays
            coords = np.empty([axes.ndim, nlines], dtype=np.int16)
            data = np.empty(nlines, dtype=np.int16)

            # Calculate the memory usage in Gigabytes
            memory_size = ((nlines * data.itemsize)+(axes.ndim*nlines*coords.itemsize))/(1024*1024*1024)
            print(f"Estimated RAM you need to read the file : {memory_size} GB")

    
                
        else :
            nlines = nbins        
            
            # Preallocate arrays    
            data = np.empty(nlines, dtype=np.int16)

            # Calculate the memory usage in Gigabytes
            memory_size = (nlines * data.itemsize)/(1024*1024*1024)
            print(f"Estimated RAM you need to read the file : {memory_size} GB")


        # Loop
        sbin = 0

        # read the rsp file and get the bin number and counts
        with gzip.open(filename, "rt") as file:
             


            #sparse case
            if sparse :
            
                progress_bar = tqdm(file, total=nlines, desc="Progress", unit="line")
                
                for line in progress_bar:

                
                    line = line.split()

                    if len(line) == 0:
                        continue

                    key = line[0]

                    if key == 'RD':

                        b = np.array(line[1:-1], dtype=np.int16)
                        c = int(line[-1])

                        coords[:, sbin] = b
                        data[sbin] = c

                        sbin += 1
                    if sbin%10e6 == 0 : 
                        progress_bar.update(10e6)
            
                progress_bar.close()
                nbins_sparse = sbin

            #non sparse case
            else :
                

                 
                binLine = False 
                         
                for line in file:
                    line = line.split()
                    
                    if len(line) == 0:
                        continue
                    
                    if line[0] == "StartStream" :
                        binLine = True
                        continue
                        
                    if binLine :
                        #check we have same number of bin than values read
                        if len(line)!=nbins :
                            print("nb of bin content read ({0}) != nb of bins {1}".format(len(line),nbins))
                            sys.exit()
                        
                        for i in tqdm(range(nbins), desc="Processing", unit="bin"):
                            data[i] = line[i]  
                
                        # we reshape the bincontent to the response matrice dimension
                        # note that for non sparse matrice SigmaTau and Dist are not used
                        data = np.reshape(data,tuple(axes.nbins),order="F")

                        break
        
        print("response file read ! Now we create the histogram and weight in order to "+ 
                "get the effective area")
        # create histpy histogram

        
        if sparse :
            dr = Histogram(axes, contents=COO(coords=coords[:, :nbins_sparse], data= data[:nbins_sparse], shape = tuple(axes.nbins)))

        else :
        
            dr = Histogram(axes, contents=data)
        
        
        
	
	
        # Weight to get effective area

        ewidth = dr.axes['Ei'].widths
        ecenters = dr.axes['Ei'].centers
        
        #print(ewidth)
        #print(ecenters)

        if Spectrumfile is not None and norm=="file":
            print("normalisation : spectrum file")
            # From spectrum file
            spec = pd.read_csv(Spectrumfile, sep=" ")
            spec = spec.iloc[:-1]
            hspec = Histogram([h_spec.axes[1]])
            hspec[:] = np.interp(hspec.axis.centers,
                             spec.iloc[:, 0].to_numpy(),
                             spec.iloc[:, 1].to_numpy(),
                             left=0,
                             right=0) * ewidth
            hspec /= np.sum(hspec)

            nperchannel_norm = hspec[:]

        elif norm=="powerlaw":
            print("normalisation : powerlaw with index {0} with energy range [{1}-{2}]keV".format(alpha,emin,emax))
            # From powerlaw

            e_lo = dr.axes['Ei'].lower_bounds
            e_hi = dr.axes['Ei'].upper_bounds

            e_lo = np.minimum(emax, e_lo)
            e_hi = np.minimum(emax, e_hi)

            e_lo = np.maximum(emin, e_lo)
            e_hi = np.maximum(emin, e_hi)

            if alpha == 1:

                nperchannel_norm = np.log(e_hi/e_low) / np.log(emax/emin)
                
            else:

                a = 1 - alpha
                
                nperchannel_norm = (e_hi**a - e_lo**a) / (emax**a - emin**a)            

        elif norm =="Linear" :
            print("normalisation : linear with energy range [{0}-{1}]".format(emin,emax))
            nperchannel_norm = ewidth / (emax-emin)
            
        elif norm=="Mono" :
            print("normalisation : mono")

            nperchannel_norm = np.array([1.])
            
        nperchannel = nperchannel_norm * nevents_sim
        # Full-sky?
        if not single_pixel:

            # Assumming all FISBEL pixels have the same area
            nperchannel /= dr.axes['NuLambda'].nbins

        # Area
        counts2area = area_sim / nperchannel
        dr_area = dr * dr.expand_dims(counts2area, 'Ei')

        

        #delete the array of data in order to release some memory
        del data 


        # end of weight now we create the .h5 structure

        npix = dr_area.axes['NuLambda'].nbins

        # remove the .h5 file if it already exist
        try:
            os.remove(filename.replace(".rsp.gz", "_nside{0}.area.h5".format(nside)))
        except:
            pass

        # create a .h5 file with the good structure
        filename = filename.replace(
        ".rsp.gz", "_nside{0}.area.h5".format(nside))
        
        f = h5.File(filename, mode='w')

        drm = f.create_group('DRM')

        # Header
        drm.attrs['UNIT'] = 'cm2'

        #sparse
        if sparse :
            drm.attrs['SPARSE'] = True
            
             # Axes
            axes = drm.create_group('AXES', track_order=True)

            for axis in dr.axes[['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi','SigmaTau','Dist']]:

                axis_dataset = axes.create_dataset(axis.label,
                                           data=axis.edges)
                                           

                if axis.label in ['NuLambda', 'PsiChi','SigmaTau']:

                    # HEALPix
                    axis_dataset.attrs['TYPE'] = 'healpix'

                    axis_dataset.attrs['NSIDE'] = nside

                    axis_dataset.attrs['SCHEME'] = 'ring'

                else:

                    # 1D
                    axis_dataset.attrs['TYPE'] = axis.axis_scale

                    if axis.label in ['Ei', 'Em']:
                        axis_dataset.attrs['UNIT'] = 'keV'
                        axis_dataset.attrs['TYPE'] = 'log'
                    elif axis.label in ['Phi']:
                        axis_dataset.attrs['UNIT'] = 'deg'
                        axis_dataset.attrs['TYPE'] = 'linear'
                    elif axis.label in ['Dist']:
                        axis_dataset.attrs['UNIT'] = 'cm'
                        axis_dataset.attrs['TYPE'] = 'linear'
                    else:
                        raise ValueError("Shouldn't happend")

                axis_description = {'Ei': "Initial simulated energy",
                            'NuLambda': "Location of the simulated source in the spacecraft coordinates",
                            'Em': "Measured energy",
                            'Phi': "Compton angle",
                            'PsiChi': "Location in the Compton Data Space",
                            'SigmaTau': "Electron recoil angle",
                            'Dist': "Distance from first interaction"
                            }

                axis_dataset.attrs['DESCRIPTION'] = axis_description[axis.label]
    
        #non sparse    
        else :
            drm.attrs['SPARSE'] = False            

            # Axes
            axes = drm.create_group('AXES', track_order=True)

            #keep the same dimension order of the data
            for axis in dr.axes[['NuLambda','Ei', 'Em', 'Phi', 'PsiChi']]:#'SigmaTau','Dist']]:

                axis_dataset = axes.create_dataset(axis.label,
                                           data=axis.edges)
                                           

                if axis.label in ['NuLambda', 'PsiChi']:#,'SigmaTau']:

                    # HEALPix
                    axis_dataset.attrs['TYPE'] = 'healpix'

                    axis_dataset.attrs['NSIDE'] = nside

                    axis_dataset.attrs['SCHEME'] = 'ring'
    
                else:

                    # 1D
                    axis_dataset.attrs['TYPE'] = axis.axis_scale

                    if axis.label in ['Ei', 'Em']:
                        axis_dataset.attrs['UNIT'] = 'keV'
                        axis_dataset.attrs['TYPE'] = 'log'
                    elif axis.label in ['Phi']:
                        axis_dataset.attrs['UNIT'] = 'deg'
                        axis_dataset.attrs['TYPE'] = 'linear'
                        #elif axis.label in ['Dist']:
                        #    axis_dataset.attrs['UNIT'] = 'cm'
                        #    axis_dataset.attrs['TYPE'] = 'linear'
                    else:
                        raise ValueError("Shouldn't happend")

                axis_description = {'Ei': "Initial simulated energy",
                            'NuLambda': "Location of the simulated source in the spacecraft coordinates",
                            'Em': "Measured energy",
                            'Phi': "Compton angle",
                            'PsiChi': "Location in the Compton Data Space",
                            #'SigmaTau': "Electron recoil angle",
                            #'Dist': "Distance from first interaction"
                            }

                axis_dataset.attrs['DESCRIPTION'] = axis_description[axis.label]
       


        #sparse matrice
        if sparse :
        
            progress_bar = tqdm(total=npix, desc="Progress", unit="nbpixel")
            # Contents. Sparse arrays
            coords = drm.create_dataset('BIN_NUMBERS',
                                (npix,),
                                dtype=h5.vlen_dtype(int),
                                compression="gzip")

            data = drm.create_dataset('CONTENTS',
                              (npix,),
                              dtype=h5.vlen_dtype(float),
                              compression="gzip")
        

        


            for b in range(npix):
        
                #print(f"{b}/{npix}")
        
                pix_slice = dr_area[{'NuLambda':b}]
                
        
                coords[b] = pix_slice.coords.flatten()
                data[b] = pix_slice.data
                progress_bar.update(1)
            
            progress_bar.close()

        #non sparse
        else :
           
     
            data = drm.create_dataset('CONTENTS',
                              data=np.transpose(dr_area.contents, axes = [1,0,2,3,4]),
                              
                              compression="gzip")
        
            
                
                
        

        
        #close the .h5 file in write mode in order to reopen it in read mode after
        f.close()
        
        new = cls(filename)

        new._file = h5.File(filename, mode='r')
        new._drm = new._file['DRM']

        new._unit = u.Unit(new._drm.attrs['UNIT'])
        new._sparse = new._drm.attrs['SPARSE']
        

        # Axes
        axes = []

        for axis_label in new._drm["AXES"]:

            axis = new._drm['AXES'][axis_label]

            axis_type = axis.attrs['TYPE']



            if axis_type == 'healpix':

                axes += [HealpixAxis(edges=np.array(axis),
                                         nside=axis.attrs['NSIDE'],
                                         label=axis_label,
                                         scheme=axis.attrs['SCHEME'],
                                         coordsys=SpacecraftFrame())]

            else:
                axes += [Axis(np.array(axis) * u.Unit(axis.attrs['UNIT']),
                                  scale=axis_type,
                                  label=axis_label)]

                

        new._axes = Axes(axes)

        # Init HealpixMap (local coordinates, main axis)
        HealpixBase.__init__(new,
                                 base=new.axes['NuLambda'],
                                 coordsys=SpacecraftFrame())

        return new

    @property
    def is_sparse(self):
        return self._sparse

    @property
    def ndim(self):
        """
        Dimensionality of detector response matrix.

        Returns
        -------
        int
        """

        return self.axes.ndim

    @property
    def axes(self):
        """
        List of axes.

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        return self._axes

    @property
    def unit(self):
        """
        Physical unit of the contents of the detector reponse.

        Returns
        -------
        :py:class:`astropy.units.Unit`
        """

        return self._unit

    def __getitem__(self, pix):

        if not isinstance(pix, (int, np.integer)) or pix < 0 or not pix < self.npix:
            raise IndexError("Pixel number out of range, or not an integer")

        #check if we have sparse matrice or not

        if self._sparse:
            coords = np.reshape(
                self._file['DRM']['BIN_NUMBERS'][pix], (self.ndim-1, -1))
            data = np.array(self._file['DRM']['CONTENTS'][pix])

            return DetectorResponse(self.axes[1:],
                                contents=COO(coords=coords,
                                             data=data,
                                             shape=tuple(self.axes.nbins[1:])),
                                unit=self.unit)
                                
        else :
            data = self._file['DRM']['CONTENTS'][pix]
            return DetectorResponse(self.axes[1:],
                                contents=data, unit=self.unit)                 

    def close(self):
        """
        Close the HDF5 file containing the response
        """

        self._file.close()

    def __enter__(self):
        """
        Start a context manager
        """

        return self

    def __exit__(self, type, value, traceback):
        """
        Exit a context manager
        """

        self.close()

    @property
    def filename(self):
        """
        Path to on-disk file containing DetectorResponse

        Returns
        -------
        :py:class:`~pathlib.Path`
        """

        return Path(self._file.filename)

    def get_interp_response(self, coord):
        """
        Get the bilinearly interpolated response at a given coordinate location.

        Parameters
        ----------
        coord : :py:class:`astropy.coordinates.SkyCoord`
            Coordinate in the :py:class:`.SpacecraftFrame`

        Returns
        -------
        :py:class:`DetectorResponse`
        """

        pixels, weights = self.get_interp_weights(coord)

        

        dr = DetectorResponse(self.axes[1:],
                              sparse=self._sparse,
                              unit=self.unit)
        
        
        for p, w in zip(pixels, weights):

            dr += self[p]*w

        return dr

    def get_point_source_response(self,
                                  exposure_map = None,
                                  coord = None,
                                  scatt_map = None):
        """
        Convolve the all-sky detector response with exposure for a source at a given
        sky location.

        Provide either a exposure map (aka dweel time map) or a combination of a 
        sky coordinate and a spacecraft attitude map.

        Parameters
        ----------
        exposure_map : :py:class:`mhealpy.HealpixMap`
            Effective time spent by the source at each pixel location in spacecraft coordinates
        coord : :py:class:`astropy.coordinates.SkyCoord`
            Source coordinate
        scatt_map : :py:class:`SpacecraftAttitudeMap`
            Spacecraft attitude map
        
        Returns
        -------
        :py:class:`PointSourceResponse`
        """

        # TODO: deprecate exposure_map in favor of coords + scatt map for both local
        # and interntial coords
        
        if exposure_map is not None:
            if not self.conformable(exposure_map):
                raise ValueError(
                    "Exposure map has a different grid than the detector response")

            psr = PointSourceResponse(self.axes[1:],
                                      sparse=self._sparse,
                                      unit=u.cm*u.cm*u.s)

            for p in range(self.npix):

                if exposure_map[p] != 0:
                    psr += self[p]*exposure_map[p]

            return psr

        else:

            # Rotate to inertial coordinates

            if coord is None or scatt_map is None:
                raise ValueError("Provide either exposure map or coord + scatt_map")
            
            if isinstance(coord.frame, SpacecraftFrame):
                raise ValueError("Local coordinate + scatt_map not currently supported")

            if self.is_sparse:
                raise ValueError("Coord +  scatt_map currently only supported for dense responses")

            axis = "PsiChi"

            coords_axis = Axis(np.arange(coord.size+1), label = 'coords')

            psr = Histogram([coords_axis] + list(deepcopy(self.axes[1:])), 
                            unit = self.unit * scatt_map.unit)
            
            psr.axes[axis].coordsys = coord.frame

            for i,(pixels, exposure) in \
                enumerate(zip(scatt_map.contents.coords.transpose(),
                              scatt_map.contents.data)):

                #gc.collect() # HDF5 cache issues
                
                att = Attitude.from_axes(x = scatt_map.axes['x'].pix2skycoord(pixels[0]),
                                         y = scatt_map.axes['y'].pix2skycoord(pixels[1]))

                coord.attitude = att

                #TODO: Change this to interpolation
                loc_nulambda_pixels = np.array(self.axes['NuLambda'].find_bin(coord),
                                               ndmin = 1)
                
                dr_pix = Histogram.concatenate(coords_axis, [self[i] for i in loc_nulambda_pixels])

                dr_pix.axes['PsiChi'].coordsys = SpacecraftFrame(attitude = att)

                self._sum_rot_hist(dr_pix, psr, exposure)

            # Convert to PSR
            psr = tuple([PointSourceResponse(psr.axes[1:],
                                             contents = data,
                                             sparse = psr.is_sparse,
                                             unit = psr.unit)
                         for data in psr[:]])
            
            if coord.size == 1:
                return psr[0]
            else:
                return psr
            
    @staticmethod
    def _sum_rot_hist(h, h_new, exposure, axis = "PsiChi"):
        """
        Rotate a histogram with HealpixAxis h into the grid of h_new, and sum
        it up with the weight of exposure.

        Meant to rotate the PsiChi of a CDS from local to galactic
        """
        
        axis_id = h.axes.label_to_index(axis)

        old_axes = h.axes
        new_axes = h_new.axes

        old_axis = h.axes[axis_id]
        new_axis = h_new.axes[axis_id]

        # Convolve
        # TODO: Change this to interpolation (pixels + weights)
        old_pixels = old_axis.find_bin(new_axis.pix2skycoord(np.arange(new_axis.nbins)))
        # NOTE: there are some pixels that are duplicated, since the center 2 pixels
        # of the original grid can land within the boundaries of a single pixel
        # of the target grid. The following commented code fixes this, but it's slow, and
        # the effect is very small, so maybe not worth it
        # nulambda_npix = h.axes['NuLamnda'].nbins    
        # new_norm = np.zeros(shape = nulambda_npix)
        # for p in old_pixels:
        #     h_slice = h[{axis:p}]
        #     norm_rot += np.sum(h_slice, axis = tuple(np.arange(1, h_slice.ndim)))
        # old_norm = np.sum(h, axis = tuple(np.arange(1, h.ndim)))
        # norm_corr = h.expand_dims(norm / norm_rot, "NuLambda")

        for old_pix,new_pix in zip(old_pixels,range(new_axis.npix)):

            #h_new[{axis:new_pix}] += exposure * h[{axis: old_pix}] # * norm_corr
            # The following code does the same than the code above, but is faster
            # However, it uses some internal functionality in histpy, which is bad practice
            # TODO: change this in a future version. We might need to modify histpy so that
            # this is not needed
            
            old_indices = tuple([slice(None)]*axis_id + [old_pix+1])
            new_indices = tuple([slice(None)]*axis_id + [new_pix+1])

            h_new._contents[new_indices] += exposure * h._contents[old_indices] # * norm_corr
                        

    def __str__(self):
        return f"{self.__class__.__name__}(filename = '{self.filename.resolve()}')"

    def __repr__(self):

        output = (f"FILENAME: '{self.filename.resolve()}'\n"
                  f"AXES:\n")

        for naxis, axis in enumerate(self.axes):

            if naxis == 0:
                description = "Location of the simulated source in the spacecraft coordinates"
            else:
                description = self._drm['AXES'][axis.label].attrs['DESCRIPTION']

            output += (f"  {axis.label}:\n"
                       f"    DESCRIPTION: '{description}'\n")

            if isinstance(axis, HealpixAxis):
                output += (f"    TYPE: 'healpix'\n"
                           f"    NPIX: {axis.npix}\n"
                           f"    NSIDE: {axis.nside}\n"
                           f"    SCHEME: '{axis.scheme}'\n")
            else:
                output += (f"    TYPE: '{axis.axis_scale}'\n"
                           f"    UNIT: '{axis.unit}'\n"
                           f"    NBINS: {axis.nbins}\n"
                           f"    EDGES: [{', '.join([str(e) for e in axis.edges])}]\n")

        return output

    def _repr_pretty_(self, p, cycle):

        if cycle:
            p.text(str(self))
        else:
            p.text(repr(self))


def cosi_response(argv=None):
    """
    Print the content of a detector response to stdout.
    """

    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] <command> [<args>] <filename> [<options>]
            """),
        description=textwrap.dedent(
            """
            Quick view of the information contained in a response file

            %(prog)s --help
            %(prog)s dump header [FILENAME]
            %(prog)s dump aeff [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s dump expectation [FILENAME] --config [CONFIG]
            %(prog)s plot aeff [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s plot dispersion [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s plot expectation [FILENAME] --lon [LON] --lat [LAT]

            Arguments:
            - header: Response header and axes information
            - aeff: Effective area
            - dispersion: Energy dispection matrix
            - expectation: Expected number of counts
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('command',
                      help=argparse.SUPPRESS)
    apar.add_argument('args', nargs='*',
                      help=argparse.SUPPRESS)
    apar.add_argument('filename',
                      help="Path to instrument response")
    apar.add_argument('--lon',
                      help="Longitude in sopacecraft coordinates. e.g. '11deg'")
    apar.add_argument('--lat',
                      help="Latitude in sopacecraft coordinates. e.g. '10deg'")
    apar.add_argument('--output', '-o',
                      help="Save output to file. Default: stdout")
    apar.add_argument('--config', '-c',
                      help="Path to config file describing exposure and source charateristics.")
    apar.add_argument('--config-override', dest='override',
                      help="Override option in config file")

    args = apar.parse_args(argv)

    # Config
    if args.config is None:
        config = Configurator()
    else:
        config = Configurator.open(args.config)

        if args.override is not None:
            config.override(args.override)

    # Get info
    with FullDetectorResponse.open(args.filename) as response:

        # Commands and functions
        def get_drm():

            lat = Quantity(args.lat)
            lon = Quantity(args.lon)

            loc = SkyCoord(lon=lon, lat=lat, frame=SpacecraftFrame())

            return response.get_interp_response(loc)

        def get_expectation():

            # Exposure map
            exposure_map = HealpixMap(base=response,
                                      unit=u.s,
                                      coordsys=SpacecraftFrame())

            ti = Time(config['exposure:time_i'])
            tf = Time(config['exposure:time_f'])
            dt = (tf-ti).to(u.s)

            exposure_map[:4] = dt/4

            logger.warning(f"Spacecraft file not yet implemented, faking source on "
                           f"axis from {ti} to {tf} ({dt:.2f})")

            # Point source response
            psr = response.get_point_source_response(exposure_map)

            # Spectrum
            model = ModelParser(model_dict=config['sources']).get_model()

            for src_name, src in model.point_sources.items():
                for comp_name, component in src.components.items():
                    logger.info(f"Using spectrum:\n {component.shape}")

            # Expectation
            expectation = psr.get_expectation(spectrum).project('Em')

            return expectation

        def command_dump():

            if len(args.args) != 1:
                apar.error("Command 'dump' takes a single argument")

            option = args.args[0]

            if option == 'header':

                result = repr(response)

            elif option == 'aeff':

                drm = get_drm()

                aeff = drm.get_spectral_response().get_effective_area()

                result = "#Energy[keV]     Aeff[cm2]\n"

                for e, a in zip(aeff.axis.centers, aeff):
                    # IMC: fix this latter when histpy has units
                    result += f"{e.to_value(u.keV):>12.2e}  {a.to_value(u.cm*u.cm):>12.2e}\n"

            elif option == 'expectation':

                expectation = get_expectation()

                result = "#Energy_min[keV]   Energy_max[keV]  Expected_counts\n"

                for emin, emax, ex in zip(expectation.axis.lower_bounds,
                                          expectation.axis.upper_bounds,
                                          expectation):
                    # IMC: fix this latter when histpy has units
                    result += (f"{emin.to_value(u.keV):>16.2e}  "
                               f"{emax.to_value(u.keV):>16.2e}  "
                               f"{ex:>15.2e}\n")

            else:

                apar.error(f"Argument '{option}' not valid for 'dump' command")

            if args.output is None:
                print(result)
            else:
                logger.info(f"Saving result to {Path(args.output).resolve()}")
                f = open(args.output, 'a')
                f.write(result)
                f.close()

        def command_plot():

            if len(args.args) != 1:
                apar.error("Command 'plot' takes a single argument")

            option = args.args[0]

            if option == 'aeff':

                drm = get_drm()

                drm.get_spectral_response().get_effective_area().plot(errorbars=False)

            elif option == 'dispersion':

                drm = get_drm()

                drm.get_spectral_response().get_dispersion_matrix().plot()

            elif option == 'expectation':

                expectation = get_expectation().plot(errorbars=False)

            else:

                apar.error(f"Argument '{option}' not valid for 'plot' command")

            if args.output is None:
                plt.show()
            else:
                logger.info(f"Saving plot to {Path(args.output).resolve()}")
                plt.savefig(args.output)

        # Run
        if args.command == 'plot':
            command_plot()
        elif args.command == 'dump':
            command_dump()
        else:
            apar.error(f"Command '{args.command}' unknown")
