# Imports:
from cosipy import UnBinnedData
from cosipy import test_data
import os
import shutil
import gzip
import numpy as np
import pytest

def test_unbinned_data_all(tmp_path):

    # Load file with dataIO class.
    # Note: this is the first 10 seconds of the crab sim from mini-DC2.
    yaml = os.path.join(test_data.path,"inputs_crab.yaml")
    analysis = UnBinnedData(yaml)
    test_filename = analysis.data_file
    analysis.data_file = os.path.join(test_data.path,analysis.data_file)
    analysis.ori_file = os.path.join(test_data.path,analysis.ori_file)
  
    # Test pointing information:

    # Read tra file, getting pointing info from ori file.
    # Also test writing output to h5 file.
    analysis.unbinned_output = "hdf5"
    analysis.read_tra(use_ori=True,output_name=tmp_path/"test_h5")
    xpointings_ori = analysis.cosi_dataset['Xpointings (glon,glat)']
    ypointings_ori = analysis.cosi_dataset['Ypointings (glon,glat)']
    zpointings_ori = analysis.cosi_dataset['Zpointings (glon,glat)']

    # Read tra file, getting pointing info from tra file.
    # Also test writing output to fits file.
    analysis.unbinned_output = "fits"
    analysis.read_tra(use_ori=False,output_name=tmp_path/"test_fits")
    xpointings_tra = analysis.cosi_dataset['Xpointings (glon,glat)']
    ypointings_tra = analysis.cosi_dataset['Ypointings (glon,glat)']
    zpointings_tra = analysis.cosi_dataset['Zpointings (glon,glat)']

    # Compare:
    xpointings_diff = np.absolute(xpointings_ori - xpointings_tra)
    ypointings_diff = np.absolute(ypointings_ori - ypointings_tra)
    zpointings_diff = np.absolute(zpointings_ori - zpointings_tra)
    
    # We'll use a tolerance of 1e-3 radians,
    # which is 0.057 degrees (3.44 arcminutes).
    # Note: This is an acceptable tolerance for an ori file with
    # a 1s cadence, but it may need to be adjusted for longer intervals. 
    check_x = xpointings_diff[xpointings_diff > 1e-3] 
    check_y = ypointings_diff[ypointings_diff > 1e-3]
    check_z = zpointings_diff[zpointings_diff > 1e-3]
    assert len(check_x) == 0
    assert len(check_y) == 0
    assert len(check_z) == 0

    # Get total number of events for other tests:
    n_events = len(analysis.cosi_dataset['TimeTags'])

    # Test chunking:
    analysis.read_tra(event_min=2, event_max=5)
    assert len(analysis.cosi_dataset['TimeTags']) == 3

    # Test file type:
    f = open(tmp_path/"test.txt","w")
    f.write("temp")
    f.close()
    analysis.data_file = "test.txt"
    with pytest.raises(SystemExit) as pytest_wrapped_exp:
        analysis.read_tra()
    assert pytest_wrapped_exp.type == SystemExit
    analysis.data_file = os.path.join(test_data.path,test_filename)
   
    # Test reading in .tra file (instead of .tra.gz):
    gz_test_file = tmp_path/test_filename
    shutil.copy(analysis.data_file, gz_test_file)

    guzip_test_file = os.path.join(tmp_path,"GalacticScan.inc1.id1.crab10sec.extracted.testsample.tra")
    with gzip.open(gz_test_file, 'rb') as f_in:
        with open(guzip_test_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    analysis.data_file = guzip_test_file
    analysis.read_tra()
    analysis.data_file = os.path.join(test_data.path,test_filename)

    # Test combine method.
    # Also test reading in fits file:
    analysis.unbinned_output = "fits"
    analysis.combine_unbinned_data([tmp_path/"test_fits.fits.gz",tmp_path/"test_fits.fits.gz"]\
            ,output_name=tmp_path/"temp_test_file")
    assert len(analysis.cosi_dataset['TimeTags']) == 2*n_events
    
    # Test time selection method.
    # Make time selection, taking just the first second. 
    # Also test reading in the hdf5 file:
    analysis.unbinned_output = "hdf5"
    analysis.tmax = 1835478001.0
    analysis.select_data_time(unbinned_data=tmp_path/"test_h5.hdf5",output_name=tmp_path/"temp_test_file")
    assert np.amax(analysis.cosi_dataset['TimeTags']) <= 1835478001.0    

    # Test energy selection method.
    analysis.select_data_energy(200,300,unbinned_data=tmp_path/"test_h5.hdf5")
    assert np.amax(analysis.cosi_dataset['Energies']) < 300
    assert np.amin(analysis.cosi_dataset['Energies']) >= 200

    # Test Compton sequence selection method
    analysis.select_data_COseq(3,4,unbinned_data=tmp_path/"test_h5.hdf5")
    assert np.amax(analysis.cosi_dataset['Compton Seq']) < 4
    assert np.amin(analysis.cosi_dataset['Compton Seq']) >= 3

    # Test SAA cut:
    analysis.cut_SAA_events(unbinned_data=tmp_path/"test_h5.hdf5")

    return

def test_unbinned_data_nopointings(tmp_path):

    from scoords import Attitude
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Load file with dataIO class.
    # Note: this is the first 10 seconds of the crab sim from mini-DC2.
    yaml = os.path.join(test_data.path,"inputs_crab.yaml")
    analysis = UnBinnedData(yaml)
    analysis.data_file = os.path.join(test_data.path,
                                      "GalacticScan.inc1.id1.crab10sec.extracted.testsample.nopointinginfo.tra.gz")
    analysis.ori_file = os.path.join(test_data.path,analysis.ori_file)


    # Read tra file, getting pointing info from ori file.
    analysis.unbinned_output = "hdf5"
    analysis.read_tra(use_ori=True)
    xpointings_ori = analysis.cosi_dataset['Xpointings (glon,glat)']
    ypointings_ori = analysis.cosi_dataset['Ypointings (glon,glat)']
    zpointings_ori = analysis.cosi_dataset['Zpointings (glon,glat)']

    # Test reading tra with no pointing info and external orientation file
    analysis.read_tra(sc_orientation=analysis.ori_file)
    xpointings_tra = analysis.cosi_dataset['Xpointings (glon,glat)']
    ypointings_tra = analysis.cosi_dataset['Ypointings (glon,glat)']
    zpointings_tra = analysis.cosi_dataset['Zpointings (glon,glat)']
    
    # Compare:
    xpointings_diff = np.absolute(xpointings_ori - xpointings_tra)
    ypointings_diff = np.absolute(ypointings_ori - ypointings_tra)
    zpointings_diff = np.absolute(zpointings_ori - zpointings_tra)
    
    # We'll use a tolerance of 1e-3 radians,
    # which is 0.057 degrees (3.44 arcminutes).
    # Note: This is an acceptable tolerance for an ori file with
    # a 1s cadence, but it may need to be adjusted for longer intervals. 
    check_x = xpointings_diff[xpointings_diff > 1e-3] 
    check_y = ypointings_diff[ypointings_diff > 1e-3]
    check_z = zpointings_diff[zpointings_diff > 1e-3]
    assert len(check_x) == 0
    assert len(check_y) == 0
    assert len(check_z) == 0

    # Test reading tra with no pointing info and an arbitrary Attitude
    att = Attitude.from_axes(SkyCoord(l=0,b=0,unit=u.deg,frame="galactic"),
                             SkyCoord(l=0,b=90,unit=u.deg,frame="galactic"),
                             frame="galactic")
    analysis.read_tra(sc_orientation=att)

    att = Attitude.from_axes(SkyCoord(l=[0],b=[0],unit=u.deg,frame="galactic"),
                             SkyCoord(l=[0],b=[90],unit=u.deg,frame="galactic"),
                             frame="galactic")
    analysis.read_tra(sc_orientation=att)

    with pytest.raises(ValueError):
        att = Attitude.from_axes(SkyCoord(l=[0,0],b=[0,90],unit=u.deg,frame="galactic"),
                                 SkyCoord(l=[0,0],b=[90,45],unit=u.deg,frame="galactic"),
                                 frame="galactic")
        analysis.read_tra(sc_orientation=att)
            
    # Test reading tra with no pointing info and no other info
    with pytest.raises(ValueError):
        analysis.data_file = os.path.join(test_data.path,
            "GalacticScan.inc1.id1.crab10sec.extracted.testsample.nopointinginfo.tra.gz")
        analysis.read_tra()
