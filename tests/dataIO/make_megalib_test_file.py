# Imports:
from cosipy import ReadTraTest
from cosipy import test_data
import os

# This file is only needed if you want to make a new test file
# with MEGAlib, i.e. parse the tra file with the MEGAlib event reader.
# It's kept here for convenience.

yaml = os.path.join(test_data.path,"inputs_crab.yaml")
analysis = ReadTraTest(yaml)
analysis.data_file = os.path.join(test_data.path,analysis.data_file)
analysis.read_tra_old(make_plots=False)
