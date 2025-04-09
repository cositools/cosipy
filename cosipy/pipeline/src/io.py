from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftFile

def get_binned_data(yaml_path, udata_path, bdata_name, psichi_coo):
    """
    Creates a binned dataset from a yaml file and an unbinned data file.
    Args:
        psichi_coo: either "galactic" or "local"

    """
    data=BinnedData(yaml_path)
    data.get_binned_data(unbinned_data=udata_path, output_name=bdata_name, psichi_binning=psichi_coo,make_binning_plots=False)
    return data


def load_binned_data(yaml_path,data_path):
    """
    Loads a BinnedData instance from an hdf5
    """
    data = BinnedData(yaml_path)
    data.load_binned_data_from_hdf5(data_path)
    return data.binned_data

def load_ori(ori_path):
    """
    Loads an orientation file
    """
    ori = SpacecraftFile.parse_from_file(ori_path)
    return ori