# Imports:
from cosipy import UnBinnedData
from cosipy import test_data
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# For comparing dataIO calculation to MEGAlib:
def compare(original,new,title,make_plots=False):

    diff = (original - new)

    if make_plots == True:
        plt.plot(diff,ls="",marker='o')
        plt.xlabel("Event")
        plt.ylabel("original - new")
        plt.title(title)
        plt.savefig("%s.pdf" %title)
        plt.show()
        plt.close()

    return diff

# Read tra file with dataIO class.
# Note: this is the GRB sim from mini-DC2. 
yaml = os.path.join(test_data.path,"inputs_GRB.yaml")
analysis = UnBinnedData(yaml)
analysis.data_file = os.path.join(test_data.path,analysis.data_file)
analysis.read_tra(output_name="GRB_unbinned_data",run_test=True)
os.system("rm GRB_unbinned_data.hdf5")

# Read in MEGAlib's calculations from its event reader:
mega_data = os.path.join(test_data.path,"unbinned_data_MEGAlib_calc.hdf5")
dict_old = analysis.get_dict_from_hdf5(mega_data)
energy_old = dict_old["Energies"]
time_old = dict_old["TimeTags"]
phi_old = dict_old["Phi"]
dist_old = dict_old["Distance"]
lonX_old = dict_old["Xpointings"].T[0]
latX_old = dict_old["Xpointings"].T[1]
lonZ_old = dict_old["Zpointings"].T[0]
latZ_old = dict_old["Zpointings"].T[1]
lonY_old = dict_old["Ypointings"].T[0]
latY_old = dict_old["Ypointings"].T[1]
chi_loc_old = dict_old['Chi local']
psi_loc_old = dict_old['Psi local']
chi_gal_old = dict_old['Chi galactic']
psi_gal_old = dict_old['Psi galactic']

# For comparing chi_loc, psi_loc=0 values are arbitrary,
# so we exclude them from the comparison.
psi_zero_index = psi_loc_old == 0

# Define dictionaries for comparing:
energies_dict = {"old":energy_old,"new":analysis.cosi_dataset["Energies"],"name":"Energies"}
time_dict = {"old":time_old,"new":analysis.cosi_dataset["TimeTags"],"name":"TimeTags"}
phi_dict = {"old":phi_old,"new":analysis.cosi_dataset["Phi"],"name":"Phi"}
dist_dict = {"old":dist_old,"new":analysis.cosi_dataset["Distance"],"name":"Distance"}
lonX_dict = {"old":lonX_old,"new":analysis.cosi_dataset["Xpointings"].T[0],"name":"lonX"}
latX_dict = {"old":latX_old,"new":analysis.cosi_dataset["Xpointings"].T[1],"name":"latX"}
lonZ_dict = {"old":lonZ_old,"new":analysis.cosi_dataset["Zpointings"].T[0],"name":"lonZ"}
latZ_dict = {"old":latZ_old,"new":analysis.cosi_dataset["Zpointings"].T[1],"name":"latZ"}
lonY_dict = {"old":lonY_old,"new":analysis.cosi_dataset["Ypointings"].T[0],"name":"lonY"}
latY_dict = {"old":latY_old,"new":analysis.cosi_dataset["Ypointings"].T[1],"name":"latY"}
chi_loc_dict = {"old":chi_loc_old[~psi_zero_index],"new":analysis.chi_loc_test[~psi_zero_index],"name":"chi_loc"}
psi_loc_dict = {"old":psi_loc_old,"new":analysis.psi_loc_test,"name":"psi_loc"}
chi_gal_dict = {"old":chi_gal_old,"new":analysis.chi_gal_test,"name":"chi_gal"}
psi_gal_dict = {"old":psi_gal_old,"new":analysis.psi_gal_test,"name":"psi_gal"}

# Make comparison:
print("Comparing to MEGAlib calculation:")
test_list = [energies_dict,time_dict,phi_dict,\
        dist_dict,lonX_dict,latX_dict,lonZ_dict,latZ_dict,lonY_dict,latY_dict,\
        chi_loc_dict,psi_loc_dict,chi_gal_dict,psi_gal_dict]
for each in test_list:
    diff = compare(each["old"],each["new"],each["name"],make_plots=False)
    if np.amax(diff) > 1e-12:
        print("ERROR: Definition does not match MEGAlib: %s" %each["name"])
        sys.exit()
    else:
        print("Passed: %s" %each["name"])
