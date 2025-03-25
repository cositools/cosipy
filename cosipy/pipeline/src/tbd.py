

def get_binned_data(yaml_path, udata_path, bdata_name, psichi_coo):
    """
    Creates a binned dataset from a yaml file and an unbinned data file.
    Args:
        psichi_coo: either "galactic" or "local"

    """
    data=BinnedData(yaml_path)
    data.get_binned_data(unbinned_data=udata_path, output_name=bdata_name, psichi_binning=psichi_coo,make_binning_plots=False)
    return data

def make_eqsize_tslices(tstart, tstop, nbins):
    """
    Takes a time interval and a number of desired bins and computes the edges
    of equal time slices
    Returns:
        tmins, tmaxs : edges
    """
    dt = (tstop - tstart) / nbins
    tmins = np.array([], dtype=float)  # Initialize as empty numpy arrays
    tmaxs = np.array([], dtype=float)  # Initialize as empty numpy arrays
    #
    for i in range(nbins):
        tmin = tstart + i * dt
        tmax = tmin + dt
        tmins = np.append(tmins, tmin)
        tmaxs = np.append(tmaxs, tmax)
    return tmins, tmaxs


def make_minsn_tslices(tstart, tstop, yaml_path, data_path, min_sn, max_slices):
    """
    Makes time slices requiring minimum total S/N

    """
    step = (tstop - tstart) / max_slices
    tmins = np.array([], dtype=float)
    tmaxs = np.array([], dtype=float)
    #
    data = load_binned_data(yaml_path, data_path)
    #
    tmax = tstart
    for i in range(max_slices):
        tmin = tmax
        tmax_i = tstart + (i + 1) * step
        #
        data_sliced = tslice_binned_data(data, tmin, tmax_i)
        signal = np.sum(data_sliced.todense().contents)
        noise = np.sqrt(signal)
        #
        sn = signal / noise
        if (sn >= min_sn and tmax_i < tstop):
            tmins = np.append(tmins, tmin)
            tmax = tmax_i
            tmaxs = np.append(tmaxs, tmax)
        elif (tmax_i == tstop):
            tmaxs[-1] = tmax_i
    return tmins, tmaxs
