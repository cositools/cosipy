from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from cosipy.util import fetch_wasabi_file
from pathlib import Path
from threeML import LinearPolarization, SpectralComponent, PointSource, Model, JointLikelihood, DataList
from scoords import SpacecraftFrame
import numpy as np

from cosipy import BinnedData, COSILike
from cosipy.spacecraftfile import SpacecraftFile
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = BinnedData(test_data.path / 'polarization_data_mlm.yaml')
analysis.load_binned_data_from_hdf5(test_data.path / 'polarization_data_binned.hdf5')
response_path = test_data.path / 'test_polarization_response.h5'
sc_orientation = SpacecraftFile.open(test_data.path / 'polarization_ori.fits')
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

source_direction = SkyCoord(0, 70, representation_type='spherical', frame=SpacecraftFrame(attitude=attitude), unit=u.deg).transform_to('galactic')

polarization = LinearPolarization(0.5, 100)
spectral_component = SpectralComponent('test', spectrum, polarization)

source = PointSource('test',
                     l = source_direction.l.deg,
                     b = source_direction.b.deg,
                     components = [spectral_component])

source.components['test'].shape.K.fix = True
source.components['test'].shape.E0.fix = True
source.components['test'].shape.alpha.fix = True
source.components['test'].shape.beta.fix = True

model = Model(source)

def test_polarization_fit():

    cosi = COSILike("cosi",
                    dr = response_path,
                    data = analysis.binned_data.project('Em', 'Phi', 'PsiChi'),
                    bkg = analysis.binned_data.project('Em', 'Phi', 'PsiChi') * 0.,
                    sc_orientation = sc_orientation,
                    response_pa_convention = 'RelativeZ')

    plugins = DataList(cosi)

    like = JointLikelihood(model, plugins, verbose=False)

    like.fit()

    assert np.allclose([source.spectrum.test.polarization.degree.value, source.spectrum.test.polarization.angle.value],
                       [0.000012, 100.], atol=[0.1, 1.0])
