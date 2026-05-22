#
import os
import subprocess
from pathlib import Path
from cosipy.util import fetch_wasabi_file
from cosipy import BinnedData
#
indir=Path.cwd() # Current path by default
#
#
#Get Response from the develop folder in wasabi (new version)
#
filename = "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"
fetch_wasabi_file('COSI-SMEX/DC4/Data/Responses/' + filename,
                  output=indir/filename,
                  checksum = '7121f094be50e7bfe9b31e53015b0e85')

#
#Get Orientation files
#
filename="DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
fetch_wasabi_file('COSI-SMEX/DC4/Data/Orientation/'+filename,
                  output=indir/filename,
                  checksum = '1b851c042acf4c909798e2401e9d2e38')

#
#Get Galactic background
#
filename='GalTotal_SA100_F98_3months_unbinned_data_filtered_with_SAAcut.fits.gz'
fetch_wasabi_file('COSI-SMEX/DC3/Data/Backgrounds/Ge/'+filename,
                  output=indir/filename,
                  checksum = '824e67875da23a42307ec13ba784147d',
		  unzip=True)

#
#Get GRB source data
#GRB
#
filename="GRB_bn081207680_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
fetch_wasabi_file('COSI-SMEX/DC3/Data/Sources/'+filename,
                  output=indir/filename,
                  checksum = '9de69d22cc880ce144a298004bb294f2',
		  unzip=True)
#
#
#==================================
#
#Combine grb and galactic background, to have a dataset for the fit
#
grb=BinnedData("bin_grbdc3.yaml")
#
grb_bk=os.path.join (indir,"galbk_grbdc3")
#
grb.combine_unbinned_data(["GRB_bn081207680_3months_unbinned_data_filtered_with_SAAcut.fits","GalTotal_SA100_F98_3months_unbinned_data_filtered_with_SAAcut.fits"], output_name=grb_bk)
subprocess.run(["gunzip","-f", "galbk_grbdc3.fits.gz"])

