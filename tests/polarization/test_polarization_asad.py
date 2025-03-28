import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationASAD
from cosipy.polarization.conventions import IAUPolarizationConvention, MEGAlibRelativeZ
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import UnBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = UnBinnedData(test_data.path / 'polarization_data.yaml')
data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')
response_path = test_data.path / 'test_polarization_response_dense.h5'
sc_orientation = SpacecraftFile.parse_from_file(test_data.path / 'polarization_ori.ori')
attitude = sc_orientation.get_attitude()[0]

a = 10. * u.keV
b = 10000. * u.keV
alpha = -1.
beta = -2.
ebreak = 350. * u.keV
K = 50. / u.cm / u.cm / u.s
spectrum = Band_Eflux(a = a.value,
                      b = b.value,
                      alpha = alpha,
                      beta = beta,
                      E0 = ebreak.value,
                      K = K.value)
spectrum.a.unit = a.unit
spectrum.b.unit = b.unit
spectrum.E0.unit = ebreak.unit
spectrum.K.unit = K.unit

source_direction = SkyCoord(0, 70, representation_type='spherical', frame=SpacecraftFrame(attitude=attitude), unit=u.deg)

bin_edges = Angle(np.linspace(-np.pi, np.pi, 10), unit=u.rad)

background = {'Psi local': [0, 0], 'Chi local': [0, 0], 'Psi galactic': [0, 0], 'Chi galactic': [0, 0], 'Energies': [300., 300.], 'TimeTags': [1., 2.]}

def test_spacecraft_fit():

    polarization_spacecraft = PolarizationASAD(source_direction, spectrum, bin_edges, data, background, sc_orientation, response_path, response_convention='RelativeZ', show_plots=True, fit_convention=MEGAlibRelativeZ(attitude=attitude))

    polarization_fit_spacecraft = polarization_spacecraft.fit(show_plots=True)

    assert np.allclose([polarization_fit_spacecraft['fraction'], polarization_fit_spacecraft['fraction uncertainty'], 
                        polarization_fit_spacecraft['angle'].angle.rad, polarization_fit_spacecraft['angle uncertainty'].rad],
                        [13.73038868282377, 2.1295224814008353, 1.4851296518928818, 0.07562763316088744], atol=[1.0, 0.5, 1.0, 0.1])

def test_icrs_fit():

    polarization_icrs = PolarizationASAD(source_direction.transform_to('galactic'), spectrum, bin_edges, data, background, sc_orientation, response_path, response_convention='RelativeZ', show_plots=True)

    polarization_fit_icrs = polarization_icrs.fit(show_plots=True)

    assert np.allclose([polarization_fit_icrs['fraction'], polarization_fit_icrs['fraction uncertainty'], 
                        polarization_fit_icrs['angle'].angle.rad, polarization_fit_icrs['angle uncertainty'].rad],
                        [2.057120422245168, 0.6877456532374626, 1.4377475471600978, 0.13124860832618374], atol=[1.0, 0.5, 1.0, 0.1])
