from cosipy import test_data
from cosipy.threeml.custom_functions import SpecFromDat
import numpy as np


def test_SpecFromDat():

    f = SpecFromDat(dat = test_data.path / "test_SpecFromDat.dat")

    # test_SpecFromDat.dat is a power law with index -2 from 100 keV to 10 MeV
    assert np.all(np.isclose(f.evaluate(np.array([200,2000]), 1),
                             np.array([200.,2000.])**-2 / (1/100 - 1/10000),
                             rtol = .01
                             )
                  )
