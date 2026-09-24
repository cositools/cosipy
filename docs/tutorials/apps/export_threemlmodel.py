import astropy.units as u
from threeML import Band, PointSource, Model
from cosipy import Band_Eflux
from astromodels import Parameter
#
l = 177.42
b = -9.41
#
alpha = -1                                       
beta = -2
xp = 300. * u.keV
piv = 500. * u.keV
K = 0.1 / u.cm / u.cm / u.s / u.keV
#
spectrum = Band()
#
spectrum.beta.min_value = -5.0
spectrum.alpha.value = alpha
spectrum.beta.value = beta
spectrum.xp.value = xp.value
spectrum.K.value = K.value
spectrum.piv.value = piv.value
spectrum.xp.unit = xp.unit
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

source = PointSource("source",                     # Name of source (arbitrary, but needs to be unique)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

#
model = Model(source)                              # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)
model.save("my_model.yaml")
