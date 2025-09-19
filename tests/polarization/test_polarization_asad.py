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
response_path = test_data.path / 'test_polarization_response.h5'
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

    polarization_spacecraft = PolarizationASAD(source_direction, spectrum, bin_edges, data, background, sc_orientation, response_path, response_convention='RelativeZ', show_plots=False, fit_convention=MEGAlibRelativeZ(attitude=attitude))

    polarization_fit_spacecraft = polarization_spacecraft.fit(show_plots=False)

    assert np.allclose([polarization_fit_spacecraft['fraction'], polarization_fit_spacecraft['fraction uncertainty'],
                        polarization_fit_spacecraft['angle'].angle.rad, polarization_fit_spacecraft['angle uncertainty'].rad],
                       [0.8114804627334942, 0.8081587949263002, 1.5713378840593466, 0.5340212799099183], atol=[1.0, 0.5, 1.0, 0.1])

def test_icrs_fit():

    polarization_icrs = PolarizationASAD(source_direction.transform_to('galactic'), spectrum, bin_edges, data, background, sc_orientation, response_path, response_convention='RelativeZ', show_plots=False)

    polarization_fit_icrs = polarization_icrs.fit(show_plots=False)

    assert np.allclose([polarization_fit_icrs['fraction'], polarization_fit_icrs['fraction uncertainty'],
                        polarization_fit_icrs['angle'].angle.rad, polarization_fit_icrs['angle uncertainty'].rad],
                       [1.6268965437885632, 0.9763744515967512, 1.8111679143685155, 0.40112053082203614], atol=[1.0, 0.5, 1.0, 0.1])
