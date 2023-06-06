# Imports:
from cosipy import ReadTraTest
from cosipy import test_data
import os

yaml = os.path.join(test_data.path,"inputs_GRB.yaml")
analysis = ReadTraTest(yaml)
analysis.data_file = os.path.join(test_data.path,analysis.data_file)
analysis.read_tra(output_name="GRB_unbinned_data")
os.system("rm *.hdf5")
analysis.read_tra_old(make_plots=False)
os.system("rm *.hdf5")

