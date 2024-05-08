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

def test_unbinned_data_with_MEGAlib():

    # Read tra file with dataIO class.
    # Note: this is the crab sim from DC2. 
    yaml = os.path.join(test_data.path,"inputs_crab.yaml")
    analysis = UnBinnedData(yaml)
    analysis.data_file = os.path.join(test_data.path,analysis.data_file)
    analysis.ori_file = os.path.join(test_data.path,analysis.ori_file)
    analysis.read_tra(run_test=True, use_ori=False)

    # Read in MEGAlib's calculations from its event reader:
    mega_data = os.path.join(test_data.path,"unbinned_data_MEGAlib_calc.hdf5")
    dict_old = analysis.get_dict_from_hdf5(mega_data)
    ntestsamples = analysis.cosi_dataset["Energies"].size #Using a reduced dataset for testing purposes
    energy_old = dict_old["Energies"][:ntestsamples]
    time_old = dict_old["TimeTags"][:ntestsamples]
    phi_old = dict_old["Phi"][:ntestsamples]
    dist_old = dict_old["Distance"][:ntestsamples]
    lonX_old = dict_old["Xpointings"].T[0][:ntestsamples]
    latX_old = dict_old["Xpointings"].T[1][:ntestsamples]
    lonZ_old = dict_old["Zpointings"].T[0][:ntestsamples]
    latZ_old = dict_old["Zpointings"].T[1][:ntestsamples]
    lonY_old = dict_old["Ypointings"].T[0][:ntestsamples]
    latY_old = dict_old["Ypointings"].T[1][:ntestsamples]
    chi_loc_old = dict_old['Chi local'][:ntestsamples]
    psi_loc_old = dict_old['Psi local'][:ntestsamples]
    chi_gal_old = dict_old['Chi galactic'][:ntestsamples]
    psi_gal_old = dict_old['Psi galactic'][:ntestsamples]

    # For comparing chi_loc, psi_loc=0 values are arbitrary,
    # so we exclude them from the comparison.
    psi_zero_index = psi_loc_old == 0

    # For comparing lonZ, lonX and chi_gal, we ignore values near +/- pi,
    # b/c small differences here lead to ~2pi discrepancies. 
    cutoff = 0.1 * (np.pi/180)
    lonZ_bad_index = (np.pi - lonZ_old < cutoff) | (np.pi - np.absolute(lonZ_old) < cutoff)
    lonX_bad_index = (np.pi - lonX_old < cutoff) | (np.pi - np.absolute(lonX_old) < cutoff)
    chi_gal_bad_index = (np.pi - chi_gal_old < cutoff) | (np.pi - np.absolute(chi_gal_old) < cutoff)

    # Define dictionaries for comparing:
    energies_dict = {"old":energy_old,"new":analysis.cosi_dataset["Energies"],"name":"Energies","units":"keV"}
    time_dict = {"old":time_old,"new":analysis.cosi_dataset["TimeTags"],"name":"TimeTags","units":"s"}
    phi_dict = {"old":phi_old,"new":analysis.cosi_dataset["Phi"],"name":"Phi","units":"rad"}
    dist_dict = {"old":dist_old,"new":analysis.cosi_dataset["Distance"],"name":"Distance","units":"cm"}
    lonX_dict = {"old":lonX_old[~lonX_bad_index],"new":analysis.cosi_dataset["Xpointings (glon,glat)"].T[0][~lonX_bad_index],"name":"lonX","units":"rad"}
    latX_dict = {"old":latX_old,"new":analysis.cosi_dataset["Xpointings (glon,glat)"].T[1],"name":"latX","units":"rad"}
    lonZ_dict = {"old":lonZ_old[~lonZ_bad_index],"new":analysis.cosi_dataset["Zpointings (glon,glat)"].T[0][~lonZ_bad_index],"name":"lonZ","units":"rad"}
    latZ_dict = {"old":latZ_old,"new":analysis.cosi_dataset["Zpointings (glon,glat)"].T[1],"name":"latZ","units":"rad"}
    lonY_dict = {"old":lonY_old,"new":analysis.cosi_dataset["Ypointings (glon,glat)"].T[0],"name":"lonY","units":"rad"}
    latY_dict = {"old":latY_old,"new":analysis.cosi_dataset["Ypointings (glon,glat)"].T[1],"name":"latY","units":"rad"}
    chi_loc_dict = {"old":chi_loc_old[~psi_zero_index],"new":analysis.chi_loc_test[~psi_zero_index],"name":"chi_loc","units":"rad"}
    psi_loc_dict = {"old":psi_loc_old,"new":analysis.psi_loc_test,"name":"psi_loc","units":"rad"}
    chi_gal_dict = {"old":chi_gal_old[~chi_gal_bad_index],"new":analysis.chi_gal_test[~chi_gal_bad_index],"name":"chi_gal","units":"rad"}
    psi_gal_dict = {"old":psi_gal_old,"new":analysis.psi_gal_test,"name":"psi_gal","units":"rad"}

    # Make comparison:
    print("Comparing to MEGAlib calculation:")
    test_list = [energies_dict,time_dict,phi_dict,\
            dist_dict,lonX_dict,latX_dict,lonZ_dict,latZ_dict,lonY_dict,latY_dict,\
            chi_loc_dict,psi_loc_dict,chi_gal_dict,psi_gal_dict]
    for each in test_list:
        diff = compare(each["old"],each["new"],each["name"],make_plots=False)
        thresh = 1e-10
        if np.amax(diff) > thresh:
            bad_index = diff > thresh
            len_bad = len(each["new"][bad_index])
            len_tot = len(each["new"])
            bad_frac = len_bad/len_tot
            raise Exception(f"WARNING: Definition does not match MEGAlib: {each['name']}. "
                            f"Fraction with diff > {thresh}: {bad_frac}. "
                            f"Max difference: {np.amax(np.absolute(diff))}  {each['units']}")
        else:
            print("Passed: %s" %each["name"])
