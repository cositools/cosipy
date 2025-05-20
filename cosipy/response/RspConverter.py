import logging
logger = logging.getLogger(__name__)

from pathlib import Path

import gzip

import numpy as np

from histpy import Axes, Axis, HealpixAxis

from scoords import SpacecraftFrame

import h5py as h5
import hdf5plugin


class RspConverter():
    """
    Converter between response files stored in .rsp.gz format and
    optimized HDF5 format on disk.

    Use method convert_to_h5() to convert a .rsp.gz file to .h5.

    Use method convert_to_rsp() to convert a FullDetectorResponse
    (backed by an .h5 file) to .rsp.gz.

    """

    # map from axis labels in .rsp file to
    # axis labels in HDF5 file
    axis_name_map = {
        '"Initial energy [keV]"'      : "Ei",
        '"#nu [deg]" "#lambda [deg]"' : "NuLambda",
        '"Polarization Angle [deg]"'  : "Pol",
        '"Measured energy [keV]"'     : "Em",
        '"#psi [deg]" "#chi [deg]"'   : "PsiChi",
        '"#phi [deg]"'                : "Phi",
        '"#sigma [deg]" "#tau [deg]"' : "SigmaTau",
        '"Distance [cm]"'             : "Dist"
    }

    # parameters for non-Healpix axes
    # (unit, scale)
    axis_params = {
        "Ei":       ("keV", "log"),
        "Pol":      ("deg", "linear"),
        "Em":       ("keV", "log"),
        "Phi":      ("deg", "linear"),
        "Dist":     ("cm", "linear")
    }

    # textual descriptions of each axis (used for pretty-printing)
    axis_description = {
        'Ei': "Initial simulated energy",
        'NuLambda': "Location of the simulated source in the spacecraft coordinates",
        'Pol': "Polarization angle",
        'Em': "Measured energy",
        'PsiChi': "Location in the Compton Data Space",
        'Phi': "Compton angle",
        'SigmaTau': "Electron recoil angle",
        'Dist': "Distance from first interaction"
    }

    # ordered subset of .rsp axes to keep for HDF5 response
    fd_axis_order =  ("NuLambda", "Ei", "Pol", "Em", "Phi", "PsiChi")

    # order  of axes expected in .rsp file
    rsp_axis_order = ("Ei", "NuLambda", "Pol", "Em", "Phi", "PsiChi")

    def __init__(self,
                 default_norm="Linear",
                 default_emin=90,
                 default_emax=10000,
                 alpha=0):

        """
        Parameters
        ----------

         default_norm : str
             type of normalisation, if not specified in header;
             one of {powerlaw, Mono, Linear, Gaussian}

         default_emin, default_emax : float
             emin/emax used in the simulation source file, if
             not specified in header (for linear, powerlaw
             normalization)

         alpha : int
             vlue of spectral index (for powerlaw normalization)

        """

        self.default_norm = default_norm
        self.default_emin = default_emin
        self.default_emax = default_emax
        self.alpha = alpha


    def convert_to_h5(self,
                      rsp_filename,
                      h5_filename = None,
                      overwrite = False,
                      compress = True,
                      elt_type = None):

        """
        Given a response file in .rsp.gz format, read it
        and write it out as an HDF5 file

        Parameters
        ----------
        rsp_filename: string
           name of input file (must end with .rsp.gz)
        h5_filename : string (optional)
           name of output file (should end with .h5); if not
           specified, use base name of rsp_filename with .h5 extension
        overwrite : bool
           overwrite the target filename if it exists
        compress: bool
           true iff response HDF5 file should use internal
           compression
        elt_type: numpy datatype or None
           type used to store raw event counts; if None,
           infer smallest feasible type from data (requires
           reading the .rsp file twice!)

        Returns
        -------
        name of new response file (will be same as input,
          but with extension .h5 instead of .rsp.gz.

        """

        if h5_filename is None:
            h5_filename = str(rsp_filename).replace(".rsp.gz", ".h5")

        if Path(h5_filename).exists and not overwrite:
            raise RuntimeError(f"Not overwriting existing HDF5 file {h5_filename}")

        if elt_type is None:
            elt_type = self._get_min_elt_type(rsp_filename)

        with gzip.open(rsp_filename, "rt") as f:

            axes, hdr = self._read_response_header(f)
            eff_area = self._get_eff_area(axes, hdr)

            nbins = hdr["nbins"]
            counts, axes = self._read_rsp(f, axes, nbins, elt_type)

            self._write_h5(axes, counts, eff_area, h5_filename,
                           compress=compress, headers=hdr["headers"])

        return h5_filename


    def _read_response_header(self, rsp_file):
        """
        Read the header portion of a response file and construct the
        axes of the response.  Additional header info besides the axes
        are returned in a dictionary for later use.

        On exit from this function, the file will be positioned at the
        start of the line containing bin data.

        Parameters
        ----------
        rsp_file : file handle to open .rsp.gz file

        Returns
        -------
        tuple (axes, hdr)
          axes -- Axes object containing axes specified in header
          hdr  -- dictionary of additional header information

        """

        hdr = {
            "nevents_sim" : 0,
            "norm"        : self.default_norm,
            "norm_params" : (self.default_emin, self.default_emax),
            "area_sim"    : 0,
            "nbins"       : 0,
            "headers"     : {}
        }

        axes_names = []
        axes_edges = []
        axes_types = []

        for line in rsp_file:

            line = line.split()

            if len(line) == 0 or line[0][0] == '#':
                continue # skip blanks and comments

            key = line[0]
            match key:
                case 'TS':
                    hdr["nevents_sim"] = int(line[1])
                    hdr["headers"][key] = " ".join(line[1:])

                case 'SA':
                    hdr["area_sim"] = float(line[1])
                    hdr["headers"][key] = " ".join(line[1:])

                case 'SP':
                    if len(line) > 1:
                        hdr["norm"] = str(line[1])
                        norm = hdr["norm"]

                        if norm == "Linear" :
                            # emin, emax
                            hdr["norm_params"] = ( int(line[2]), int(line[3]) )
                        elif norm == "Gaussian" :
                            # Gauss_mean, Gauss_sig, Gauss_cutoff
                            hdr["norm_params"] = ( float(line[2]), float(line[3]), float(line[4]) )

                    else:
                        logger.info(f"norm not found in file! Assuming {hdr['norm']}")
                        assert hdr['norm'] == 'Linear', "parameters not given for default norm"

                    hdr["headers"][key] = " ".join(line[1:])

                case 'MS':
                    is_sparse = (line[1] == "true")
                    if is_sparse:
                        raise RuntimeError("Not supported: sparse .rsp files")

                case 'RD': # start of data for sparse .rsp
                    raise RuntimeError("Not supported: sparse .rsp files")

                case 'AN':
                    axes_names.append(' '.join(line[1:]))

                case 'AD':
                    if axes_types[-1] == "FISBEL":
                        raise RuntimeError("FISBEL binning not currently supported")
                    elif axes_types[-1] == "HEALPix":
                        if line[2] != "RING":
                            raise RuntimeError(f"Scheme {line[2]} not supported")

                        if line[1] == '-1':   # Single-pixel axis, i.e., all-sky
                            axes_edges.append((-1,'ring'))
                        else:
                            nside = int(2**int(line[1]))
                            axes_edges.append((nside, 'ring'))
                    else:
                        axes_edges.append(np.array(line[1:], dtype='float'))

                case 'AT':
                    axes_types.append(line[2])

                case 'StartStream': # start of data for dense .rsp
                    hdr["nbins"] = int(line[1])
                    break

                case 'StopStream': # end of data for dense .rsp -- should never appear
                    raise RunTimeError("StopStream encountered before StartStream")

                case _: # any other field
                    hdr["headers"][key] = " ".join(line[1:])

        # check if the type of spectrum is known
        assert hdr["norm"] in ("powerlaw", "Mono", "Linear", "Gaussian"), \
            f"unknown normalisation {hdr['norm']}"
        logger.info(f"normalisation is {hdr['norm']}")

        # check the number of simulated events is not 0
        assert hdr["nevents_sim"] != 0, \
            "number of simulated events is 0"

        # check that we are ready to start consuming bin values
        assert hdr["nbins"] > 0, \
            "no bin count provided for response"


        axes_labels = [ RspConverter.axis_name_map[n] for n in axes_names ]

        # Construct Axes object from specified axes' properties
        axes = []
        for axis_edges, axis_type, axis_label in zip(axes_edges, axes_types, axes_labels):

            # skip axes that are not in HDF5 axis order; we assume that
            # these axes are *not* dimeisions of the counts data!
            if axis_label not in RspConverter.fd_axis_order:
                continue

            if axis_type == 'HEALPix':
                nside, scheme = axis_edges
                if nside == -1: # Single bin axis -- i.e., all-sky
                    nside = 1
                    edges = (0,1)
                else:
                    edges = None

                    axes.append(HealpixAxis(edges=edges,
                                            nside=nside,
                                            scheme=scheme,
                                            coordsys=SpacecraftFrame(),
                                            label=axis_label))
            else:
                unit, scale = RspConverter.axis_params[axis_label]
                axes.append(Axis(edges=axis_edges, unit=unit, scale=scale, label=axis_label))

        axes = Axes(axes, copy_axes = False)

        return (axes, hdr)


    def _get_eff_area(self, axes, hdr):
        """
        Compute the effective area correction to the raw counts in the response.

        Parameters
        ----------
        axes : Axes object
           axes of response
        hdr : dict
           dictionary of additional response info read from header

        Returns
        -------
        eff_area : ndarray of float
           effective area for each Ei bin

        """

        ewidth = axes['Ei'].widths
        ecenters = axes['Ei'].centers

        norm = hdr["norm"]

        # If we have one single bin, treat the Gaussian norm like the mono one.
        # Also check that the Gaussian spectrum is fully contained in that bin
        if norm == "Gaussian" and len(ewidth) == 1:

            from scipy.special import erf

            Gauss_mean = hdr["norm_params"][0]

            edges = axes['Ei'].edges
            gauss_int = \
                0.5 * (1 + erf( (edges[0] - Gauss_mean)/(4*np.sqrt(2)) ) ) + \
                0.5 * (1 + erf( (edges[1] - Gauss_mean)/(4*np.sqrt(2)) ) )

            assert gauss_int == 1, "The gaussian spectrum is not fully contained in this single bin!"
            logger.info("Only one bin so we will use the Mono normalisation")

            norm = "Mono"

        match norm:

            case "Linear":

                emin, emax = hdr["norm_params"]

                logger.info(f"normalisation: linear with energy range [{emin}-{emax}]")

                nperchannel_norm = ewidth / (emax - emin)

            case "Mono" :
                logger.info("normalisation: mono")

                nperchannel_norm = np.array([1.])

            case "powerlaw":
                emin, emax = hdr["norm_params"]

                logger.info(f"normalisation: powerlaw with index {self.alpha} with energy range [{emin}-{emax}]keV")
                # From powerlaw
                e_lo = axes['Ei'].lower_bounds
                e_hi = axes['Ei'].upper_bounds

                e_lo = np.minimum(emax, e_lo)
                e_hi = np.minimum(emax, e_hi)

                e_lo = np.maximum(emin, e_lo)
                e_hi = np.maximum(emin, e_hi)

                if self.alpha == 1:
                    nperchannel_norm = np.log(e_hi/e_low) / np.log(emax/emin)
                else:
                    a = 1 - self.alpha
                    nperchannel_norm = (e_hi**a - e_lo**a) / (emax**a - emin**a)

            case "Gaussian" :
                raise NotImplementedError("Gaussian norm for multiple bins not yet implemented")

        # If Nulambda is full-sky, its nbins will be 1, so division is a no-op.
        # We assume all FISBEL pixels have the same area.
        nperchannel = nperchannel_norm * hdr["nevents_sim"] / axes["NuLambda"].nbins

        # Area
        eff_area = hdr["area_sim"] / nperchannel
        return eff_area


    @staticmethod
    def _get_min_elt_type(rsp_filename):
        """
        Determine the smallest integer type sufficient
        to hold every count a dense response file.

        Parameters
        ----------
        rsp_filename : string
           response file name

        Returns
        -------
        numpy dtype -- type of width sufficient to hold all counts

        """

        with gzip.open(rsp_filename, "rt") as rsp_file:

            for line in rsp_file:
                # consume the file header
                line = line.split()
                if len(line) > 0 and line[0] == "StartStream":
                    break

            # read all the counts and keep the maximum
            b = BufferedList(rsp_file)
            vmax = np.uint64(0)
            while not b.eol():
                vals = np.fromstring(b.read(), dtype=np.uint64, sep=' ')
                vmax = np.maximum(vmax, np.max(vals))

        t = np.min_scalar_type(vmax)

        logger.info(f"Using element type {t} for response counts")

        return t


    @staticmethod
    def _read_rsp(rsp_file, axes, nbins, elt_type):
        """
        Read a dense response with nbins bins from file rsp_file.
        Create the response matrix as type elt_type, which must
        be large enough to represent every value.

        We assume that rsp_file is positioned at the beginning
        of the count data (i.e., the header has already been read).

        In addition to reading the raw counts, we convert them from
        the F order specified in the .rsp file to the C order required
        by HDF5 and permute them so that the NuLambda axis comes
        first, followed by the Ei axis. The remaining axes are ordered
        as in the .rsp file.

        NB: reading the raw counts requires nbins * sizeof(elt_type)
        bytes of RAM; F to C conversion temporarily *doubles* this
        space requirement.

        Parameters
        ----------
        rsp_file : file handle
           open .rsp.gz file
        nbins : int
           number of bins to read from file
        axes : Axes object
           axes for response
        elt_type : numpy dtype
           type of integer used to hold counts

        Returns
        -------
        tuple (counts, axes)
          counts -- array of CDS counts, with shape matching axes.shape
          axes -- reordered axis list, matching counts

        """

        counts = np.empty(nbins, dtype=elt_type)
        b = BufferedList(rsp_file)

        ptr = 0
        while not b.eol():
            vals = np.fromstring(b.read(), dtype=np.uint64, sep=' ')
            if np.any(vals > np.iinfo(counts.dtype).max):
                raise ValueError("Count value out of range for type {counts.dtype}")
            counts[ptr:ptr + len(vals)] = vals
            ptr += len(vals)

        if ptr < nbins:
            raise ValueError("Not enough values read from file")

        counts = np.reshape(counts, axes.shape, order='F')

        # reorder the axes as specified for the HDF5 file
        ax_order = [ ax for ax in RspConverter.fd_axis_order
                     if ax in axes.labels ]
        idx_order = axes.label_to_index(ax_order)
        axes = axes[idx_order]

        # reorder counts dimension to match reordered axes
        counts = np.transpose(counts, idx_order)

        # make sure the counts array is contiguous and C-oidered
        # (this is very expensive if the .rsp is F-ordered, but
        # if we don't do it, h5py will do it less efficiently when
        # we write the output.)
        counts = np.ascontiguousarray(counts)

        return counts, axes


    @staticmethod
    def _write_h5(axes, counts, eff_area, h5_filename,
                  compress=True, headers=None):
        """
        Write the response to an HDF5 file.  All data is stored in a
        group "DRM" within the file.

        The response is stored as raw counts (in whatever bit width
        was chosen during .rsp reading) in the CONTENTS dataset,
        together with an effective area array, of length equal to the
        number of Ei bins, in the EFF_AREA dataset.

        The full floating-point response is therefore obtained by
        performing

          counts * axes.expand_dims(eff_area, axes.label_to_index('Ei'))

        The CONTENTS dataset is chunked according to the "good chunks"
        scheme, with one chunk per (NuLambda, Ei, Pol) (i.e., one
        chunk per CDS).  Chunks are compressed using a scheme with
        much lower read-time overhead than gzip.

        The AXES group describes the axes of the response; it should
        be read using Axes.open() to create an Axes object.  There is
        also an AXIS_DESCRIPTIONS group whose attributes are brief
        textual descriptions of each named axis.

        If headers is not None, it is a dictionary of header keys, each
        giving the contents of the line with that key in the .rsp file.
        These key/contents pairs are stored as attributes of a HEADERS
        group.

        Parameters
        ----------
        axes : Axes object
          axes of response
        counts : ndarray of int of shape axes.shape
          raw count data
        eff_area : ndarray of float
          effective area scaling for each Ei
        h5_filename : string
          file name to be written
        compress : bool
          True iff HDF5 file should use internal compression
        headers : dict or None
          dictionary of keys and contents of header lines other than axis info,
          to be written to the HEADERS group
        """

        with h5.File(h5_filename, mode="w") as f:

            drm = f.create_group('DRM')

            if headers is not None:
                # save any header values not deducible from Axes or contents
                header_group = drm.create_group('HEADERS', track_order=True)
                for key in headers:
                    header_group.attrs[key] = headers[key]

            drm.attrs['UNIT'] = 'cm2'
            drm.attrs['SPARSE'] = False

            axes_group = drm.create_group('AXES', track_order=True)
            axes.write(axes_group)

            axes_desc_group = drm.create_group('AXIS_DESCRIPTIONS')
            for label in axes.labels:
                axes_desc_group.attrs[label] = RspConverter.axis_description[label]

            # save effective area for each Ei; make it an array if scalar
            eff_area = np.broadcast_to(eff_area, axes["Ei"].nbins)
            drm.create_dataset('EFF_AREA', data=eff_area)

            # These are the recommended chunk sizes per axis according
            # to the COSI "good chunks" notebook.  Basically, we store
            # a chunk representing the CDS for each of the possible
            # source params.
            chunk_sizes = [ 1 if axis.label in ("NuLambda", "Ei", "Pol")
                            else axis.nbins
                            for axis in axes ]

            if compress:
                compression = hdf5plugin.Bitshuffle()
                shuffle = (counts.dtype.itemsize > 1)
            else:
                compression = None
                shuffle = False

            ds = drm.create_dataset('CONTENTS',
                                    counts.shape,
                                    dtype=counts.dtype,
                                    chunks=tuple(chunk_sizes),
                                    compression=compression,
                                    shuffle=shuffle)

            # write output for one bin on axis 0 at a time,
            # which should reduce transient memory usage to
            # at most a small fraction of the number of counts.
            # (Axis 0 is NuLambda, which generally has many bins.)
            for j in range(axes[0].nbins):
                ds[j] = counts[j]


    @staticmethod
    def convert_to_rsp(fullDetectorResponse,
                       rsp_filename,
                       overwrite = False):
        """
        Convert a FullDetectorResponse object backed by an HDF5 file
        into a textual .rsp.gz response.  We reuse the header
        information stored in the HDF5 file, along with its axes and
        counts.

        Parameters
        ----------
        fullDetectorResponse : FullDetectorResponse
           object to be converted
        rsp_filename : string
           path to write .rsp.gz file (should end with .rsp.gz)
        overwrite : bool
           if true, overwrite existing response if it exists

        """

        if Path(rsp_filename).exists and not overwrite:
            raise RuntimeError(f"Not overwriting existing .rsp.gz file {rsp_filename}")

        # reorder axes if needed to match the expected order for an .rsp file
        axes = fullDetectorResponse._axes
        ax_order = [ ax for ax in RspConverter.rsp_axis_order
                     if ax in axes.labels ]
        idx_order = axes.label_to_index(ax_order)

        axes = axes[idx_order]

        counts = np.array(fullDetectorResponse._drm["CONTENTS"])
        counts = np.transpose(counts, idx_order)

        RspConverter._write_rsp(fullDetectorResponse._drm["HEADERS"].attrs,
                                axes, counts, rsp_filename)


    @staticmethod
    def _write_rsp(headers, axes, counts, rsp_filename):
        """
        Write an .rsp.gz file with all necessary info.

        Parameters
        ----------
        headers : dict-like
          stored headers from original response
        axes : Axes
          axes to write
        counts :
          counts of Histogram
        rsp_filename :
           name of response file to write (should be .rsp.gz).

        """

        # invert the axis name map to get names to write to .rsp
        axis_names = {}
        for desc in RspConverter.axis_name_map:
            axis_names[RspConverter.axis_name_map[desc]] = desc

        with gzip.open(rsp_filename, "wt") as f:

            f.write("# computed reduced response\n")

            # write non-axis headers
            for key in headers:
                f.write(f"{key} {headers[key]}\n")

            # write axes
            for axis in axes:
                f.write(f"AN {axis_names[axis.label]}\n")

                if isinstance(axis, HealpixAxis):
                    f.write("AT 2D HEALPix\n")
                    if axis.nbins == 1:
                        order = -1
                    else:
                        order = int(np.log2(axis.nside))

                    f.write(f"AD {order} RING\n")

                else:
                    f.write(f"AT 1D BinEdges\n")

                    edges = np.array_str(axis.edges.value,
                                         max_line_width=1000000000).strip("[] ")
                    f.write(f"AD {edges}\n")

            # write counts

            f.write(f"StartStream {counts.size}\n")

            # counts are printed one one line in FORTRAN order
            cts = np.ravel(counts, order='F')
            chunksize = 10000000
            s = 0
            while s < len(cts):
                vals = cts[s:s+chunksize]
                s += len(vals)
                f.write("".join((f"{str(v)} " for v in vals)))
                f.write("\n")

            f.write("StopStream\n")


class BufferedList:
    """
    A reader for a list of values from a file that uses
    bounded memory.

    The input is assumed to be space-separated integer values
    on a single line.

    Parameters
    ----------
       f -- a text file (must support read() with bounded size,
            but not seek or other operations)
       bufsize -- maximum amount of text to read at once
       sep -- separator character between successive values

    """

    def __init__(self, f, bufsize=10000000, sep=' '):
        self.f = f
        self.sep = sep

        self.bufsize = bufsize
        self.buf = f.readline(bufsize)

    def eol(self):
        """
        Return true iff all subsequent calls to read() will return
        the empty string.
        """

        return (self.buf == '')

    def read(self):
        """
        Read a buffer full of separated values and return it as a string.
        Ensure that we never break the string in the middle of a value.

        If we have reached the end of line, all subsequent calls to read()
        will return the empty string.
        """

        s = self.buf

        if len(s) > 0:
            if s[-1] == '\n': # EOL found -- trim it; no more data
                s = s[:-1]
                self.buf = ''
            else:
                # get next buffer full of data
                self.buf = self.f.readline(self.bufsize)

                if s[-1] != self.sep:
                    # last buffer did not end with separator; read to
                    # next separator and move those chars to s.
                    loc = self.buf.find(self.sep)
                    if loc != -1:
                        s += self.buf[:loc]
                        self.buf = self.buf[loc+1:]

        return s
