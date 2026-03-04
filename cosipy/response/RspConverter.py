import logging
logger = logging.getLogger(__name__)

from pathlib import Path

import gzip

import numpy as np

from histpy import Axes, Axis, HealpixAxis

from scoords import SpacecraftFrame

import h5py as h5
import hdf5plugin

from tqdm.autonotebook import tqdm


class RspConverter():
    """
    Converter between response files stored in .rsp format and
    optimized HDF5 format on disk.

    Use method convert_to_h5() to convert a .rsp file to .h5.

    Use method convert_to_rsp() to convert a FullDetectorResponse
    (backed by an .h5 file) to .rsp.

    To read and write compressed .rsp files, add ".gz" to the end of
    the .rsp filename.

    """

    # version of response file format
    rsp_version = 2

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
                 norm=None,
                 norm_params=[],
                 quiet=False,
                 bufsize = 10000000):

        """Parameters
        ----------
        norm : str, optional
             type of normalization to use to compute E_i weighting
             for effective area correction if none specified in .rsp
             file header; one of {powerlaw, Mono, Linear,
             Gaussian}. If not specified, use Linear norm as default.
        norm_params : list-like, optional
             parameters of default normalization.  If not specified,
             use values [90, 10000] for Linear normalization for
             compatibility with old API.
        quiet : boolean
             disable logging and progress bars (default False)
        bufsize: int
             rough size of buffer to be used for reading/writing counts

        """

        self.quiet = quiet
        self.bufsize = bufsize

        if norm is None:
            self.norm = "Linear"
            self.norm_params = [90, 10000]
        else:
            self.norm = norm
            self.norm_params = norm_params

    @staticmethod
    def _open_rsp(rsp_filename, mode):
        """
        Open an .rsp file with or without .gz extension.  If .gz,
        use gzip to open; otherwise, open as a regular file.
        The resulting file object can be used in a context manager.

        Parameters
        ----------
        rsp_filename : Path or string
          name of file
        mode : str
          mode in which to open file

        Returns
        -------
        open file or gzip stream object

        """

        rsp_path = Path(rsp_filename)

        if rsp_path.suffix == ".gz":
            return gzip.open(rsp_filename, mode)
        else:
            return open(rsp_filename, mode)

    def convert_to_h5(self,
                      rsp_filename,
                      h5_filename = None,
                      overwrite = False,
                      compress = True,
                      elt_type = None):

        """
        Given a response file in .rsp format, read it
        and write it out as an HDF5 file

        Parameters
        ----------
        rsp_filename: string
           name of input file (must end with .rsp or .rsp.gz)
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
        string : name of output file

        """

        if h5_filename is None:
            rsp_path = Path(rsp_filename)

            # strip any .rsp and .gz from end of path and add .h5
            h5_path = rsp_path.parent / rsp_path.stem
            while h5_path.suffix in {".rsp", "gz"}:
                h5_path = h5_path.with_suffix("")

            h5_filename = h5_path.parent / (h5_path.stem + ".h5")

        if Path(h5_filename).exists() and not overwrite:
            raise RuntimeError(f"Not overwriting existing HDF5 file {h5_filename}")

        # read all info from the .rsp file
        with self._open_rsp(rsp_filename, "rt") as f:

            axes, hdr = self._read_response_header(f)
            eff_area = self._get_eff_area_correction(axes, hdr)

            if elt_type is None:
                elt_type = self._get_min_elt_type(rsp_filename)

            nbins = hdr["nbins"]
            counts = self._read_counts(f, axes, nbins, elt_type)

        # reorder the axes as specified for the HDF5 file
        ax_order = [ ax for ax in RspConverter.fd_axis_order
                     if ax in axes.labels ]
        idx_order = axes.label_to_index(ax_order)
        axes = axes[idx_order]

        # create the output file and determine how many initial axes
        # constitute the index of each chunk
        h5_file, n_idx_axes = \
            self._create_h5(axes, elt_type, eff_area, h5_filename,
                            compress=compress, headers=hdr["headers"])

        # transpose counts into the output axis order
        counts = counts.transpose(idx_order)

        # write the counts to the HDF5 file
        self._write_counts(counts, n_idx_axes, h5_file)

        h5_file.close()

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
        rsp_file : file handle to open .rsp file

        Returns
        -------
        tuple (axes, hdr)
          axes -- Axes object containing axes specified in header
          hdr  -- dictionary of additional header information

        """

        hdr = {
            "nevents_sim" : 0,
            "norm"        : None,
            "norm_params" : [],
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
                    # norm information is only useful if non-empty
                    if len(line) > 1:
                        norm = str(line[1])
                        norm_params = self._validate_norm_params(norm, line[2:])

                        hdr["norm"] = norm
                        hdr["norm_params"] = norm_params

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
                    raise RuntimeError("StopStream encountered before StartStream")

                case _: # any other field
                    hdr["headers"][key] = " ".join(line[1:])


        # if no spectral normalization was specified, use the provided
        # default
        if hdr["norm"] is None:
            logger.warning("RSP file does not specify spectral normalization; "
                           f"using default: {self.norm} {' '.join(str(x) for x in self.norm_params)}")

            hdr["norm"] = self.norm
            hdr["norm_params"] = self._validate_norm_params(self.norm, self.norm_params)

            # add a synthetic SP header matching the default normalization
            if len(hdr["norm_params"]) > 0:
                param_str = " " + " ".join(str(x) for x in hdr['norm_params'])
            else:
                param_str = ""

            hdr["headers"]["SP"] = f"{self.norm}{param_str}"

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

    def _validate_norm_params(self, norm, params):
        """
        Validate parameters for a spectral normalization specification.

        Parameters
        ----------
        norm : str
           name of the normaliztion scheme
        params : list-like
           parameters for normalization.  These may be either strings
           or other types; they will be converted to the correct types
           for their scheme.

        Returns
        -------
        params : tuple
           parsed parameters of types appropriate for scheme

        """

        match norm:
            case 'Mono':
                if len(params) > 1:
                    raise ValueError(f"Mono normalization takes a most one param; {len(params)} given")
                # emono if given, else dummy value
                params = () if len(params) == 0 else ( int(params[0]), )

            case 'Linear':
                if len(params) != 2:
                    raise ValueError(f"Linear normalization takes two params; {len(params)} given")
                # emin, emax
                params = ( int(params[0]), int(params[1]) )

            case 'Gaussian':
                if len(params) != 3:
                    raise ValueError(f"Gaussian normalization takes three params; {len(params)} given")
                # mean, sig, cutoff
                params = ( float(params[0]), float(params[1]), float(params[2]) )

            case 'powerlaw':
                if len(params) != 3:
                    raise ValueError(f"powerlaw normalization takes three params; {len(params)} given")
                # emin, emax, alpha
                params = ( int(params[0]), int(params[1]), float(params[2]) )

            case _:
                raise ValueError(f"Unknown normalization {norm}; must be one of Mono, Linear, Gaussian, or powerlaw")

        return params

    def _get_eff_area_correction(self, axes, hdr):
        """
        Compute the effective area correction to the raw counts in the
        response.

        The area correction is defined as follows: for a given source
        direction s and energy E_i, let n_i be the number of incident
        particles generated by the simulation, and let a_i be the area
        of the region in the plane perpendicular to s over which they
        are generated. (This region covers the entire physical area of
        the detector.)  If the detector generates Compton events for
        n_c of incident photons, then the effective area is

          A(s, E_i) = a_i n_c(s, E_i)/n_i(s, E_i).

        Because we store raw (integer) event counts, the count matrix
        already includes the factor n_c. Hence, the area *correction*
        A'(s, E_i) is simply a_i / n_i(s, E_i).

        We assume that the simulation used to construct the response
        casts an equal number of photons for every source direction s,
        and that for a given direction, the fraction of photons
        cast in a given energy bin E_i is given by f(E_i) derived from
        the E_i normalization rule used for the simulation.  Hence, if
        n_i is the *total* number of incident photons cast, and M is the
        number of distinct source directions, we have

          A'(s, E_i) = a_i / (n_i f(E_i)/M)

        This quantity is the same for each s, so we return an array of
        size equal to the number of E_i bins.

        Parameters
        ----------
        axes : Axes object
           axes of response
        hdr : dict
           dictionary of additional response info read from header

        Returns
        -------
        eff_area : ndarray of float
           effective area correction for each Ei bin

        """

        def gauss_int(x, mu, sigma):
            from scipy.special import erf
            z = (x - mu)/(sigma * np.sqrt(2))
            return 0.5*(1 + erf(z))

        norm = hdr["norm"]
        params = hdr["norm_params"]

        ei_axis = axes['Ei']
        e_lo = ei_axis.lower_bounds.value
        e_hi = ei_axis.upper_bounds.value

        match norm:

            case "Linear":
                emin, emax = params

                if not self.quiet:
                    logger.info(f"normalization: linear with energy range [{emin}-{emax}]")

                e_lo = np.minimum(emax, e_lo)
                e_hi = np.minimum(emax, e_hi)

                e_lo = np.maximum(emin, e_lo)
                e_hi = np.maximum(emin, e_hi)

                nperchannel_norm = (e_hi - e_lo) / (emax - emin)

            case "Mono" :
                if not self.quiet:
                    logger.info("normalization: mono")

                if params == ():
                    if ei_axis.nbins > 1:
                        raise ValueError("Cannot specify Mono norm without energy for Ei axis with multiple bins")
                    else:
                        # all energy is in the single bin
                        nperchannel_norm = np.array([1.])

                        # fill in a header value so we don't write "None"
                        hdr["norm_params"] = (ei_axis.centers[0],)
                else:
                    emono = params[0]

                    # set just the Ei bin containing the mono energy to 1
                    nperchannel_norm = np.zeros(ei_axis.nbins)
                    nperchannel_norm[(e_lo <= emono) & (e_hi >= emono)] = 1.

            case "powerlaw":
                emin, emax, alpha = params

                if not self.quiet:
                    logger.info(f"normalization: powerlaw with index {alpha} with energy range [{emin}-{emax}]keV")

                e_lo = np.minimum(emax, e_lo)
                e_hi = np.minimum(emax, e_hi)

                e_lo = np.maximum(emin, e_lo)
                e_hi = np.maximum(emin, e_hi)

                if alpha == 1:
                    nperchannel_norm = np.log(e_hi/e_lo) / np.log(emax/emin)
                else:
                    a = 1 - alpha
                    nperchannel_norm = (e_hi**a - e_lo**a) / (emax**a - emin**a)

            case "Gaussian" :
                mean, sdev, cutoff = params

                emin = mean - cutoff * sdev
                emax = mean + cutoff * sdev

                logger.info(f"normalization: Gaussian with energy range [{emin}-{emax}]keV")

                e_lo = np.minimum(emax, e_lo)
                e_hi = np.minimum(emax, e_hi)

                e_lo = np.maximum(emin, e_lo)
                e_hi = np.maximum(emin, e_hi)

                nperchannel_norm = (gauss_int(e_hi, mean, sdev) - gauss_int(e_lo, mean, sdev)) / (emax - emin)

        # If Nulambda is full-sky, its nbins will be 1, so division is a no-op.
        # We assume all FISBEL pixels have the same area.
        nperchannel = nperchannel_norm * hdr["nevents_sim"] / axes["NuLambda"].nbins

        zero_weights = (nperchannel == 0.)

        if np.any(zero_weights):
            logger.warning("Spectral normalization gives zero incident photons in some Ei bins; "
                           "eff_area correction for those bins will be set to zero!")

        # Area
        eff_area = np.zeros(len(nperchannel))
        eff_area[~zero_weights] = hdr["area_sim"] / nperchannel[~zero_weights]

        return eff_area


    @staticmethod
    def _create_h5(axes, counts_dtype, eff_area, h5_filename,
                  compress=True, headers=None):
        """
        Create the HDF5 file to hold the response, writing everything
        except the raw counts.  All data is stored in a group "DRM"
        within the file.  This group has an attribute VERSION that
        records the response format version.

        The response is stored as raw counts (in whatever bit width
        was chosen during .rsp reading) in the COUNTS dataset,
        together with an effective area correction array, of length
        equal to the number of Ei bins, in the EFF_AREA dataset.

        The full floating-point response is therefore obtained by
        performing

          counts * axes.expand_dims(eff_area, axes.label_to_index('Ei'))

        The COUNTS dataset is chunked according to the "good chunks"
        scheme, with one chunk per (NuLambda, Ei, Pol) (i.e., one
        chunk per CDS).  It is assumed that the output axis order can
        be divided into n initial axes that describe the index of a
        chunk, followed by the axes of the chunk itself.  Chunks are
        compressed using a scheme with much lower read-time overhead
        than gzip.

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
        counts_dtype : numpy dtype
          integer type of counts dataset
        eff_area : ndarray of float
          effective area correction for each Ei
        h5_filename : string
          file name to be written
        compress : bool
          True iff HDF5 file should use internal compression
        headers : dict or None
          dictionary of keys and contents of header lines other than axis info,
          to be written to the HEADERS group

        Returns
        -------
        f, n_idx_axes
          f : handle to newly created HDF5 file
          n_idx_axes : number of initial axes that index the chunk array

        """

        def inv_perm(p):
            """ invert a permutation of [0...len(p) - 1] """
            r = np.empty_like(p)
            for i, v in enumerate(p):
                r[v] = i
            return r

        f = h5.File(h5_filename, mode="w")

        drm = f.create_group('DRM')
        drm.attrs["VERSION"] = RspConverter.rsp_version

        header_group = drm.create_group('HEADERS')
        if headers is not None:
            # save any header values not deducible from Axes or contents
            for key in headers:
                header_group.attrs[key] = headers[key]

            # record how the header keys should be permuted when we
            # reread them from the file to recover the original order
            drm.attrs["HEADER_ORDER"] = inv_perm(np.argsort(list(headers.keys())))
        else:
            # no headers
            drm.attrs["HEADER_ORDER"] = np.array([])

        drm.attrs['UNIT'] = 'cm2'
        drm.attrs['SPARSE'] = False

        axes_group = drm.create_group('AXES')
        axes.write(axes_group)

        axes_desc_group = drm.create_group('AXIS_DESCRIPTIONS')
        for label in axes.labels:
            axes_desc_group.attrs[label] = RspConverter.axis_description[label]

        # save effective area correction for each Ei; make it an array
        # if scalar
        eff_area = np.broadcast_to(eff_area, axes["Ei"].nbins)
        drm.create_dataset('EFF_AREA', data=eff_area, track_times=False)

        # Compute the recommended chunk size per axis according
        # to the COSI "good chunks" notebook.  Basically, we store
        # a chunk representing the CDS for each of the possible
        # source params.

        is_idx_axis = [ axis.label in ("NuLambda", "Ei", "Pol")
                        for axis in axes ]

        # number of initial axes that index the array of chunks
        n_idx_axes = np.sum(is_idx_axis, dtype=int)

        assert all(is_idx_axis[:n_idx_axes]), \
            "Error: axes of each response chunk must be at end of counts' axis list"

        chunk_sizes = [ 1 if is_idx else axis.nbins
                        for axis, is_idx  in zip(axes, is_idx_axis) ]

        if compress:
            compression = hdf5plugin.Bitshuffle()
        else:
            compression = None

        ds = drm.create_dataset('COUNTS',
                                axes.shape,
                                dtype=counts_dtype,
                                chunks=tuple(chunk_sizes),
                                compression=compression,
                                track_times=False)

        return (f, n_idx_axes)


    def _get_min_elt_type(self, rsp_filename):
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

        with self._open_rsp(rsp_filename, "rt") as rsp_file:

            nbins = None
            for line in rsp_file:
                # consume the file header
                line = line.split()
                if len(line) > 0 and line[0] == "StartStream":
                    nbins = int(line[1])
                    break

            if nbins is None:
                ValueError("Could not find start of data in .rsp file")

            tq = tqdm(total=nbins,
                      desc="Getting type for counts",
                      disable=self.quiet)

            # read all the counts and keep the maximum
            b = BufferedList(rsp_file, bufsize=self.bufsize)
            vmax = np.uint64(0)
            while not b.eol():
                vals = np.fromstring(b.read(), dtype=np.uint64, sep=' ')
                vmax = np.maximum(vmax, np.max(vals))
                tq.update(n=len(vals))

            tq.close()

        t = np.min_scalar_type(vmax)

        if not self.quiet:
            logger.info(f"Using element type {t} for response counts")

        return t


    def _read_counts(self, rsp_file, axes, nbins, elt_type):
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
           open .rsp file
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
        b = BufferedList(rsp_file, bufsize=self.bufsize)

        tq = tqdm(total=nbins,
                  desc="Reading counts",
                  disable=self.quiet)

        ptr = 0
        while not b.eol():
            vals = np.fromstring(b.read(), dtype=np.uint64, sep=' ')
            nvals = len(vals)
            if np.any(vals > np.iinfo(counts.dtype).max):
                raise ValueError("Count value out of range for type {counts.dtype}")
            counts[ptr:ptr + nvals] = vals
            ptr += nvals
            tq.update(n=nvals)

        tq.close()

        if ptr < nbins:
            raise ValueError("Not enough values read from file")

        counts = np.reshape(counts, axes.shape, order='F')

        return counts


    def _write_counts(self, counts, n_idx_axes, h5_file):
        """
        Write the counts matrix to the output HDF5 file.  We
        write the data a chunk at a time to limit memory usage
        beyond the input counts array to a small fraction of its size.

        The counts matrix is divided into chunks, with the first
        n_idx_axes axes specifying the chunk index and the
        remainder indexing into a chunk.  counts is not guaranteed
        to be C-contiguous in memory, so we make it so one chunk
        at a time as we write the output.

        Parameters
        ----------
        counts : ndarray
          Matrix of counts to write
        n_idx_axes : int
          number of initial axes describing a chunk index
        h5_file : HDF5 file handle
          file containing counts dataset to write

        """

        ds = h5_file["DRM/COUNTS"]

        # number of chunks to write
        tq = tqdm(total=np.prod(counts.shape[:n_idx_axes]),
                  desc="Writing chunks",
                  disable=self.quiet)

        # Construct an iterator that enumerates the indices
        # of each chunk, so that we can extract chunks from the
        # counts array one at a time.

        it = np.nditer(counts,
                       flags=["multi_index"],
                       op_axes=[list(range(n_idx_axes))])

        while not it.finished:
            chunk_idx = it.multi_index

            chunk = counts[chunk_idx]
            chunk = np.ascontiguousarray(chunk) # to C order
            ds[chunk_idx] = chunk

            it.iternext()
            tq.update()

        tq.close()

    def convert_to_rsp(self,
                       fullDetectorResponse,
                       rsp_filename,
                       overwrite = False):
        """
        Convert a FullDetectorResponse object backed by an HDF5 file
        into a textual .rsp or .rsp.gz response.  We reuse the header
        information stored in the HDF5 file, along with its axes and
        counts.

        Parameters
        ----------
        fullDetectorResponse : FullDetectorResponse
           object to be converted
        rsp_filename : string
           path to write .rsp file; if extension is .rsp.gz, the file
           will be gzipped.
        overwrite : bool
           if true, overwrite existing response if it exists

        """

        if Path(rsp_filename).exists() and not overwrite:
            raise RuntimeError(f"Not overwriting existing file {rsp_filename}")

        # reorder axes if needed to match the expected order for an .rsp file
        axes = fullDetectorResponse._axes
        ax_order = [ ax for ax in RspConverter.rsp_axis_order
                     if ax in axes.labels ]
        idx_order = axes.label_to_index(ax_order)

        axes = axes[idx_order]

        counts = np.array(fullDetectorResponse._drm["COUNTS"])
        counts = np.transpose(counts, idx_order)

        hdrs = fullDetectorResponse.headers

        self._write_rsp(hdrs, axes, counts, rsp_filename)


    def _write_rsp(self, headers, axes, counts, rsp_filename):
        """
        Write an .rsp file with all necessary info.

        Parameters
        ----------
        headers : dict
          stored headers from original response
        axes : Axes
          axes to write
        counts :
          counts of Histogram
        rsp_filename :
           path to write .rsp file; if extension is .rsp.gz, the file
           will be gzipped.

        """

        # invert the axis name map to get names to write to .rsp
        axis_names = {}
        for desc in RspConverter.axis_name_map:
            axis_names[RspConverter.axis_name_map[desc]] = desc

        with self._open_rsp(rsp_filename, "wt") as f:

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
                                         max_line_width=100000000).strip("[] ")
                    f.write(f"AD {edges}\n")

            # write counts

            f.write(f"StartStream {counts.size}\n")

            # counts are printed on one line in FORTRAN order
            cts = np.ravel(counts, order='F')
            chunksize = self.bufsize//2 # shrink to allow for spaces

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
