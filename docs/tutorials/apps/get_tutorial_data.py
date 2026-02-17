#
import os
import subprocess
from pathlib import Path
from cosipy.util import fetch_wasabi_file
from cosipy import BinnedData
#
indir=str("./")
#

def get_data (wasabipath,outpath,unzip):
    if not os.path.exists(outpath):
        print ("Downloading")
        print (wasabipath)
        fetch_wasabi_file(wasabipath,output=outpath)
        if unzip==True:
            if outpath[-2:] == 'gz':
                print("Gunzipping")
                subprocess.run(["gunzip", outpath])
            elif outpath[-3:] == 'zip':
                print("Unzipping")
                subprocess.run(["unzip", outpath])
    return()


#
#Get Response from the develop folder in wasabi (new version)
#
filename='ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5'
wasabipath=os.path.join('COSI-SMEX/develop/Data/Responses',filename)
outpath=os.path.join(indir,filename)
get_data(wasabipath,outpath,False)

#
#Get Orientation files
#
filename="DC3_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
wasabipath=os.path.join('COSI-SMEX/develop/Data/Orientation',filename)
outpath=os.path.join(indir,filename)
get_data(wasabipath,outpath,False)



#
#Get Galactic background
#
filename='GalTotal_SA100_F98_3months_unbinned_data_filtered_with_SAAcut.fits.gz'
wasabipath=os.path.join('COSI-SMEX/DC3/Data/Backgrounds/Ge',filename)
outpath=os.path.join(indir,filename)
get_data(wasabipath,outpath,True)


#
#Get GRB source data
#
wasabirootpath="COSI-SMEX/DC3/Data/Sources"
#
#GRB
#
filename="GRB_bn081207680_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
wasabipath=os.path.join(wasabirootpath,filename)
outpath=os.path.join(indir,filename)
get_data(wasabipath,outpath,True)
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
subprocess.run(["gunzip", "galbk_grbdc3.fits.gz"])

#
