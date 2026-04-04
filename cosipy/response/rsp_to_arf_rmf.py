import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from histpy import Histogram

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile.spacecraft_file import SpacecraftHistory

class RspArfRmfConverter:

    def __init__(self, response:FullDetectorResponse,
                 ori:SpacecraftHistory,
                 target_coord:SkyCoord):

        self.response = response
        self.ori = ori

        self.dwell_map = self.ori.get_dwell_map(target_coord,
                                                nside = response.nside,
                                                scheme = response.scheme)

        self.arf = None
        self.rmf = None

    def get_psr_rsp(self):

        """Generates the point source response based on the response file
        and dwell obstime map.  livetime is used to find the exposure
        obstime for this observation.

        .. warning::
            RspArfRmfConverter currently only supports local spacecraft coordinate.
            Therefore, it is limited to short-duration events.

        Returns
        -------
        Ei_edges : numpy.ndarray
            The edges of the incident energy.
        Ei_lo : numpy.ndarray
            The lower edges of the incident energy.
        Ei_hi : numpy.ndarray
            The upper edges of the incident energy.
        Em_edges : numpy.ndarray
            The edges of the measured energy.
        Em_lo : numpy.ndarray
            The lower edges of the measured energy.
        Em_hi : numpy.ndarray
            The upper edges of the measured energy.
        areas : numpy.ndarray
            The effective area of each energy bin.
        matrix : numpy.ndarray
            The energy dispersion matrix.

        """

        with self.response as response:

            # get point source response
            psr = response.get_point_source_response(self.dwell_map)

            Ei_edges = response.axes['Ei'].edges.value
            # use float32 to match the requirement of the data type
            self.Ei_lo = np.float32(Ei_edges[:-1])
            self.Ei_hi = np.float32(Ei_edges[1:])

            Em_edges = response.axes['Em'].edges.value
            self.Em_lo = np.float32(Em_edges[:-1])
            self.Em_hi = np.float32(Em_edges[1:])

        # get the effective area and matrix
        logger.info("Getting the effective area ...")
        self.arf = np.float32(psr.project('Ei').to_dense().contents.value) / self.ori.livetime.to_value(u.s).sum()
        spectral_response = np.float32(psr.project(('Ei', 'Em')).to_dense().contents.value)
        self.rmf = np.float32(np.zeros((self.Ei_lo.size, self.Em_lo.size)))  # initialize the matrix

        logger.info("Getting the energy redistribution matrix ...")
        for i in range(self.Ei_lo.size):
            new_raw = spectral_response[i, :] / spectral_response[i, :].sum()
            self.rmf[i, :] = new_raw
        self.rmf = self.rmf.T

        return Ei_edges, self.Ei_lo, self.Ei_hi, \
            Em_edges, self.Em_lo, self.Em_hi, \
            self.arf, self.rmf

    def write_arf(self, out_name, overwrite = False):

        """
        Converts the point source response to an arf file that can be
        read by XSPEC.

        Parameters
        ----------
        out_name: str
            The name of the arf file to save.
        overwrite: bool
            Overwrite file if it exists.
        """

        if self.arf is None:
            self.get_psr_rsp()

        copyright_string = "  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "

        ## Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU()  # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32  # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string  # add comments

        col1_energ_lo = fits.Column(name="ENERG_LO", format="E", unit="keV", array=self.Em_lo)
        col2_energ_hi = fits.Column(name="ENERG_HI", format="E", unit="keV", array=self.Em_hi)
        col3_specresp = fits.Column(name="SPECRESP", format="E", unit="cm**2", array=self.arf)
        cols = fits.ColDefs([col1_energ_lo, col2_energ_hi, col3_specresp])  # create a ColDefs (column-definitions) object for all columns
        specresp_bintablehdu = fits.BinTableHDU.from_columns(cols)  # create a binary table HDU object

        specresp_bintablehdu.header.comments["TTYPE1"] = "label for field   1"
        specresp_bintablehdu.header.comments["TFORM1"] = "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT1"] = "physical unit of field"
        specresp_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        specresp_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        specresp_bintablehdu.header.comments["TTYPE3"] = "label for field   3"
        specresp_bintablehdu.header.comments["TFORM3"] = "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT3"] = "physical unit of field"

        specresp_bintablehdu.header["EXTNAME"] = ("SPECRESP", "name of this binary table extension")
        specresp_bintablehdu.header["TELESCOP"] = ("COSI", "mission/satellite name")
        specresp_bintablehdu.header["INSTRUME"] = ("COSI", "instrument/detector name")
        specresp_bintablehdu.header["FILTER"] = ("NONE", "filter in use")
        specresp_bintablehdu.header["HDUCLAS1"] = ("RESPONSE", "dataset relates to spectral response")
        specresp_bintablehdu.header["HDUCLAS2"] = ("SPECRESP", "extension contains an ARF")
        specresp_bintablehdu.header["HDUVERS"] = ("1.1.0", "version of format")

        new_arfhdus = fits.HDUList([primaryhdu, specresp_bintablehdu])
        new_arfhdus.writeto(out_name, overwrite=overwrite)

    def write_rmf(self, out_name, overwrite = False):

        """
        Converts the point source response to an rmf file that can be read by XSPEC.

        Parameters
        ----------
        out_name: str
            The name of the arf file to save.
        overwrite: bool
            Overwrite file if it exists.
        """

        copyright_string = "  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "

        ## Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU()  # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32  # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string  # add comments

        ## Create binary table HDU for MATRIX
        ### prepare colums
        energ_lo = []
        energ_hi = []
        n_grp = []
        f_chan = []
        n_chan = []
        matrix = []
        for i in range(len(self.Ei_lo)):
            energ_lo_temp = np.float32(self.Em_lo[i])
            energ_hi_temp = np.float32(self.Ei_hi[i])

            if self.rmf[:, i].sum() != 0:
                nz_matrix_idx = np.nonzero(self.rmf[:, i])[0]  # non-zero index for the matrix
                subsets = np.split(nz_matrix_idx, np.where(np.diff(nz_matrix_idx) != 1)[0] + 1)
                n_grp_temp = np.int16(len(subsets))
                f_chan_temp = []
                n_chan_temp = []
                matrix_temp = []
                for m in np.arange(n_grp_temp):
                    f_chan_temp += [subsets[m][0]]
                    n_chan_temp += [len(subsets[m])]
                for m in nz_matrix_idx:
                    matrix_temp += [self.rmf[:, i][m]]
                f_chan_temp = np.array(f_chan_temp, dtype=np.int16)
                n_chan_temp = np.array(n_chan_temp, dtype=np.int16)
                matrix_temp = np.array(matrix_temp, dtype=np.float32)
            else:
                n_grp_temp = np.int16(0)
                f_chan_temp = np.zeros(1, dtype=np.int16)
                n_chan_temp = np.zeros(1, dtype=np.int16)
                matrix_temp = np.zeros(1, dtype=np.float32)

            energ_lo.append(energ_lo_temp)
            energ_hi.append(energ_hi_temp)
            n_grp.append(n_grp_temp)
            f_chan.append(f_chan_temp)
            n_chan.append(n_chan_temp)
            matrix.append(matrix_temp)

        col1_energ_lo = fits.Column(name="ENERG_LO", format="E", unit="keV", array=energ_lo)
        col2_energ_hi = fits.Column(name="ENERG_HI", format="E", unit="keV", array=energ_hi)
        col3_n_grp = fits.Column(name="N_GRP", format="I", array=n_grp)
        col4_f_chan = fits.Column(name="F_CHAN", format="PI(54)", array=f_chan)
        col5_n_chan = fits.Column(name="N_CHAN", format="PI(54)", array=n_chan)
        col6_n_chan = fits.Column(name="MATRIX", format="PE(161)", array=matrix)
        cols = fits.ColDefs([col1_energ_lo, col2_energ_hi, col3_n_grp, col4_f_chan, col5_n_chan,
                             col6_n_chan])  # create a ColDefs (column-definitions) object for all columns
        matrix_bintablehdu = fits.BinTableHDU.from_columns(cols)  # create a binary table HDU object

        matrix_bintablehdu.header.comments["TTYPE1"] = "label for field   1 "
        matrix_bintablehdu.header.comments["TFORM1"] = "data format of field: 4-byte REAL"
        matrix_bintablehdu.header.comments["TUNIT1"] = "physical unit of field"
        matrix_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        matrix_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        matrix_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        matrix_bintablehdu.header.comments["TTYPE3"] = "label for field   3 "
        matrix_bintablehdu.header.comments["TFORM3"] = "data format of field: 2-byte INTEGER"
        matrix_bintablehdu.header.comments["TTYPE4"] = "label for field   4"
        matrix_bintablehdu.header.comments["TFORM4"] = "data format of field: variable length array"
        matrix_bintablehdu.header.comments["TTYPE5"] = "label for field   5"
        matrix_bintablehdu.header.comments["TFORM5"] = "data format of field: variable length array"
        matrix_bintablehdu.header.comments["TTYPE6"] = "label for field   6"
        matrix_bintablehdu.header.comments["TFORM6"] = "data format of field: variable length array"

        matrix_bintablehdu.header["EXTNAME"] = ("MATRIX", "name of this binary table extension")
        matrix_bintablehdu.header["TELESCOP"] = ("COSI", "mission/satellite name")
        matrix_bintablehdu.header["INSTRUME"] = ("COSI", "instrument/detector name")
        matrix_bintablehdu.header["FILTER"] = ("NONE", "filter in use")
        matrix_bintablehdu.header["CHANTYPE"] = ("PI", "total number of detector channels")
        matrix_bintablehdu.header["DETCHANS"] = (len(self.Em_lo), "total number of detector channels")
        matrix_bintablehdu.header["HDUCLASS"] = ("OGIP", "format conforms to OGIP standard")
        matrix_bintablehdu.header["HDUCLAS1"] = ("RESPONSE", "dataset relates to spectral response")
        matrix_bintablehdu.header["HDUCLAS2"] = ("RSP_MATRIX", "dataset is a spectral response matrix")
        matrix_bintablehdu.header["HDUVERS"] = ("1.3.0", "version of format")
        matrix_bintablehdu.header["TLMIN4"] = (0, "minimum value legally allowed in column 4")

        ## Create binary table HDU for EBOUNDS
        channels = np.arange(len(self.Em_lo), dtype=np.int16)
        e_min = np.float32(self.Em_lo)
        e_max = np.float32(self.Em_hi)

        col1_channels = fits.Column(name="CHANNEL", format="I", array=channels)
        col2_e_min = fits.Column(name="E_MIN", format="E", unit="keV", array=e_min)
        col3_e_max = fits.Column(name="E_MAX", format="E", unit="keV", array=e_max)
        cols = fits.ColDefs([col1_channels, col2_e_min, col3_e_max])
        ebounds_bintablehdu = fits.BinTableHDU.from_columns(cols)

        ebounds_bintablehdu.header.comments["TTYPE1"] = "label for field   1"
        ebounds_bintablehdu.header.comments["TFORM1"] = "data format of field: 2-byte INTEGER"
        ebounds_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        ebounds_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        ebounds_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        ebounds_bintablehdu.header.comments["TTYPE3"] = "label for field   3"
        ebounds_bintablehdu.header.comments["TFORM3"] = "data format of field: 4-byte REAL"
        ebounds_bintablehdu.header.comments["TUNIT3"] = "physical unit of field"

        ebounds_bintablehdu.header["EXTNAME"] = ("EBOUNDS", "name of this binary table extension")
        ebounds_bintablehdu.header["TELESCOP"] = ("COSI", "mission/satellite")
        ebounds_bintablehdu.header["INSTRUME"] = ("COSI", "nstrument/detector name")
        ebounds_bintablehdu.header["FILTER"] = ("NONE", "filter in use")
        ebounds_bintablehdu.header["CHANTYPE"] = ("PI", "channel type (PHA or PI)")
        ebounds_bintablehdu.header["DETCHANS"] = (len(self.Em_lo), "total number of detector channels")
        ebounds_bintablehdu.header["HDUCLASS"] = ("OGIP", "format conforms to OGIP standard")
        ebounds_bintablehdu.header["HDUCLAS1"] = ("RESPONSE", "dataset relates to spectral response")
        ebounds_bintablehdu.header["HDUCLAS2"] = ("EBOUNDS", "dataset is a spectral response matrix")
        ebounds_bintablehdu.header["HDUVERS"] = ("1.2.0", "version of format")

        new_rmfhdus = fits.HDUList([primaryhdu, matrix_bintablehdu, ebounds_bintablehdu])
        new_rmfhdus.writeto(out_name, overwrite=overwrite)

    def write_pha(self, out_name, src_counts, errors, rmf_file, arf_file, bkg_file,
                exposure_time, telescope="COSI", instrument="COSI", overwrite = False):

        """Generate the pha file that can be read by XSPEC. This file stores
        the counts info of the source.

        Parameters
        ----------
        out_name: str, Path
            Output path
        src_counts : numpy.ndarray
            The counts in each energy band. If you have src_counts
            with unit counts/kev/s, you must convert it to counts by
            multiplying it with exposure obstime and the energy band
            width.
        errors : numpy.ndarray
            The error for counts. It has the same unit requirement as
            src_counts.
        rmf_file : str, optional
            The rmf file name to be written into the pha file.
        arf_file : str, optional
            The arf file name to be written into the pha file.
        bkg_file : str, optional
            The background file name.
        exposure_time : float, optional
           The exposure obstime for this source observation.
        telescope : str, optional
            The name of the telecope (the default is "COSI").
        instrument : str, optional
            The instrument name (the default is "COSI").
        overwrite: bool
            Overwrite file if it exists.
        """

        channel_number = len(src_counts)

        # define other hardcoded inputs
        copyright_string = "  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "
        channels = np.arange(channel_number)

        # Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU()  # create an empty primary HDU
        primaryhdu.header[
            "BITPIX"] = -32  # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string  # add comments
        primaryhdu.header["TELESCOP"] = telescope  # add telescope keyword valie
        primaryhdu.header["INSTRUME"] = instrument  # add instrument keyword valie

        # Create binary table HDU
        a1 = np.array(channels, dtype="int32")  # I guess I need to convert the dtype to match the format J
        a2 = np.array(src_counts, dtype="int64")  # int32 is not enough for counts
        a3 = np.array(errors, dtype="int64")  # int32 is not enough for errors
        col1 = fits.Column(name="CHANNEL", format="J", array=a1)
        col2 = fits.Column(name="COUNTS", format="K", array=a2, unit="count")
        col3 = fits.Column(name="STAT_ERR", format="K", array=a3, unit="count")
        cols = fits.ColDefs([col1, col2, col3])  # create a ColDefs (column-definitions) object for all columns
        bintablehdu = fits.BinTableHDU.from_columns(cols)  # create a binary table HDU object

        # add other BinTableHDU hear keywords,their values, and comments
        bintablehdu.header.comments["TTYPE1"] = "label for field 1"
        bintablehdu.header.comments["TFORM1"] = "data format of field: 32-bit integer"
        bintablehdu.header.comments["TTYPE2"] = "label for field 2"
        bintablehdu.header.comments["TFORM2"] = "data format of field: 32-bit integer"
        bintablehdu.header.comments["TUNIT2"] = "physical unit of field 2"

        bintablehdu.header["EXTNAME"] = ("SPECTRUM", "name of this binary table extension")
        bintablehdu.header["TELESCOP"] = (telescope, "telescope/mission name")
        bintablehdu.header["INSTRUME"] = (instrument, "instrument/detector name")
        bintablehdu.header["FILTER"] = ("NONE", "filter type if any")
        bintablehdu.header["EXPOSURE"] = (exposure_time, "integration obstime in seconds")
        bintablehdu.header["BACKFILE"] = (bkg_file, "background filename")
        bintablehdu.header["BACKSCAL"] = (1, "background scaling factor")
        bintablehdu.header["CORRFILE"] = ("NONE", "associated correction filename")
        bintablehdu.header["CORRSCAL"] = (1, "correction file scaling factor")
        bintablehdu.header["CORRSCAL"] = (1, "correction file scaling factor")
        bintablehdu.header["RESPFILE"] = (rmf_file, "associated rmf filename")
        bintablehdu.header["ANCRFILE"] = (arf_file, "associated arf filename")
        bintablehdu.header["AREASCAL"] = (1, "area scaling factor")
        bintablehdu.header["STAT_ERR"] = (0, "statistical error specified if any")
        bintablehdu.header["SYS_ERR"] = (0, "systematic error specified if any")
        bintablehdu.header["GROUPING"] = (0, "grouping of the data has been defined if any")
        bintablehdu.header["QUALITY"] = (0, "data quality information specified")
        bintablehdu.header["HDUCLASS"] = ("OGIP", "format conforms to OGIP standard")
        bintablehdu.header["HDUCLAS1"] = ("SPECTRUM", "PHA dataset")
        bintablehdu.header["HDUVERS"] = ("1.2.1", "version of format")
        bintablehdu.header["POISSERR"] = (False, "Poissonian errors to be assumed, T as True")
        bintablehdu.header["CHANTYPE"] = ("PI", "channel type (PHA or PI)")
        bintablehdu.header["DETCHANS"] = (channel_number, "total number of detector channels")

        new_phahdus = fits.HDUList([primaryhdu, bintablehdu])
        new_phahdus.writeto(out_name, overwrite=overwrite)

    def plot_arf(self, ax = None, file_name=None):

        """Read the arf fits file, plot and save it.

        Parameters
        ----------
        ax: Optional(matpltlib.axes)
            Matplotlib axes to use for plotting. We'll create new one if not privided.
        file_name: str, optional
            The directory if the arf fits file (the default is `None`,
            which implies the file name will be read from the
            instance).

        Returns
        -------
        ax
        """

        if file_name is None:
            if self.arf is None:
                self.get_psr_rsp()

            areas = self.arf
            Em_lo = self.Em_lo
            Em_hi = self.Em_hi

        else:

            arf = fits.open(file_name)  # read file

            # SPECRESP HDU
            specresp_hdu = arf["SPECRESP"]

            areas = np.array(specresp_hdu.data["SPECRESP"])
            Em_lo = np.array(specresp_hdu.data["ENERG_LO"])
            Em_hi = np.array(specresp_hdu.data["ENERG_HI"])

        E_center = (Em_lo + Em_hi) / 2
        E_edges = np.append(Em_lo, Em_hi[-1])

        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(E_center, E_edges, weights=areas, histtype='step')

        ax.set_title("Effective area")
        ax.set_xlabel("Energy[$keV$]")
        ax.set_ylabel(r"Effective area [$cm^2$]")
        ax.set_xscale("log")

        return  ax

    def plot_rmf(self, ax = None,  file_name=None):

        """Read the rmf fits file, plot and save it.

        Parameters
        ----------
        ax: Optional(matpltlib.axes)
            Matplotlib axes to use for plotting. We'll create new one if not privided.
        file_name: str, optional
            The directory if the arf fits file (the default is `None`,
            which implies the file name will be read from the
            instance).

        Returns
        -------
        ax

        """

        if file_name is None:
            if self.arf is None:
                self.get_psr_rsp()

            rmf_matrix = self.rmf
            energy_low = self.Em_lo
            energy_high = self.Em_hi

        else:

            # Read rmf file
            rmf = fits.open(file_name)  # read file

            # Read the MATRIX extension
            matrix_ext = rmf['MATRIX']
            energy_low = matrix_ext.data["ENERG_LO"]  # energy bin lower edges for measured energies
            energy_high = matrix_ext.data["ENERG_HI"]  # energy bin higher edges for measured energies
            data = matrix_ext.data

            # Create a 2-d numpy array and store probability data into the redistribution matrix
            rmf_matrix = np.zeros(data.shape)  # create an empty matrix
            for i in range(data.shape[0]):  # i is the measured energy index, examine the matrix_ext.data rows by rows
                if data[i][5].sum() == 0:  # if the sum of probabilities is zero, then skip since there is no data at all
                    pass
                else:
                    # measured_energy_index = np.argwhere(energy_low == data[157][0])[0][0]
                    f_chan = data[i][3]  # get the starting channel of each subsets
                    n_chan = data[i][4]  # get the number of channels in each subsets
                    matrix = data[i][5]  # get the probabilities of this row (incident energy)
                    indices = []
                    for k in f_chan:
                        channels = np.arange(k, k + n_chan[np.argwhere(f_chan == k)][0][0]).tolist()  # generate the cha
                        indices += channels  # append the channels together
                    indices = np.array(indices)
                    for m in indices:
                        rmf_matrix[i][m] = matrix[np.argwhere(indices == m)[0][0]]  # write the probabilities into the empty matrix

        # plot the redistribution matrix
        xcenter = (energy_low + energy_high) / 2
        x_center_coords = np.repeat(xcenter, 10)
        y_center_coords = np.tile(xcenter, 10)
        energy_all_edges = np.append(energy_low, energy_high[-1])
        bin_edges = np.vstack((energy_all_edges, energy_all_edges))

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        h = Histogram((np.append(energy_low, energy_high[-1]), np.arange(rmf_matrix.shape[0]+1)), contents = np.transpose(rmf_matrix), copy_contents = False)
        ax,im = h.plot(ax)
        ax.set_xscale('log')
        ax.set_xlabel("Incident energy [$keV$]")
        ax.set_ylabel("Channel number")
        ax.set_title("Redistribution matrix")

        return ax