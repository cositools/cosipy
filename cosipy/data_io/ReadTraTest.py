# Import
from cosipy.data_io import UnBinnedData 
import matplotlib.pyplot as plt
import numpy as np

# Load MEGAlib into ROOT
import ROOT as M
M.gSystem.Load("$(MEGAlib)/lib/libMEGAlib.so")

# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()
    

class ReadTraTest(UnBinnedData):

    def compare(self,original,new,title):
       
        diff = (original - new) 
        plt.plot(diff,ls="",marker='o')
        plt.xlabel("Event")
        plt.ylabel("original - new")
        plt.title(title)
        plt.savefig("Images/%s.pdf" %title)
        plt.show()
        plt.close()

        return

    def read_tra_old(self):
        
        """
        Reads in MEGAlib .tra (or .tra.gz) file.
        Returns COSI dataset as a dictionary of the form:
        cosi_dataset = {'Full filename':self.data_file,
                        'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings':np.array([lonX,latX]).T,
                        'Ypointings':np.array([lonY,latY]).T,
                        'Zpointings':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal}
        
        Input (optional):
        tra_file: Name of tra file to read. Default is output from select_data method.
        """

        # tra file to use:
        tra_file = self.data_file

        # Make print statement:
        print()
        print("Read tra test: comparing to MEGAlib...")
        print()
         
        # Check if file exists:
        Reader = M.MFileEventsTra()
        if Reader.Open(M.MString(tra_file)) == False:
            print("Unable to open file %s. Aborting!" %self.data_file)
            sys.exit()

        # Initialise empty lists:
            
        # Total photon energy
        erg = []
        # Time tag in UNIX time
        tt = []
        # Event Type (0: CE; 4:PE; ...)
        et = []
        # Latitude of X direction of spacecraft
        latX = []
        # Lontitude of X direction of spacecraft
        lonX = []
        # Latitude of Z direction of spacecraft
        latZ = []
        # Longitude of Z direction of spacecraft
        lonZ = []
        # Compton scattering angle
        phi = []
        # Measured data space angle chi (azimuth direction; 0..360 deg)
        chi_loc = []
        # Measured data space angle psi (polar direction; 0..180 deg)
        psi_loc = []
        # First lever arm distance in cm
        dist = []
        # Measured gal angle chi (lon direction)
        chi_gal = []
        # Measured gal angle psi (lat direction)
        psi_gal = [] 

        # Browse through tra file, select events, and sort into corresponding list:
        # Note: The Reader class from MEGAlib knows where an event starts and ends and
        # returns the Event object which includes all information of an event.
        # Note: Here only select Compton events (will add Photo events later as optional).
        # Note: All calculations and definitions taken from:
        # /MEGAlib/src/response/src/MResponseImagingBinnedMode.cxx.

        while True:
            
            Event = Reader.GetNextEvent()
            if not Event:
                break
                
            # Total Energy:
            erg.append(Event.Ei())
            # Time tag in UNIX seconds:
            tt.append(Event.GetTime().GetAsSeconds())
            # Event type (0 = Compton, 4 = Photo):
            et.append(Event.GetEventType())
            # x axis of space craft pointing at GAL latitude:
            latX.append(Event.GetGalacticPointingXAxisLatitude())
            # x axis of space craft pointing at GAL longitude:
            lonX.append(Event.GetGalacticPointingXAxisLongitude())
            # z axis of space craft pointing at GAL latitude:
            latZ.append(Event.GetGalacticPointingZAxisLatitude())
            # z axis of space craft pointing at GAL longitude:
            lonZ.append(Event.GetGalacticPointingZAxisLongitude())    
            # Compton scattering angle:
            phi.append(Event.Phi()) 
            # Data space angle chi (azimuth):
            chi_loc.append((-Event.Dg()).Phi())
            # Data space angle psi (polar):
            psi_loc.append((-Event.Dg()).Theta())
            # Interaction length between first and second scatter in cm:
            dist.append(Event.FirstLeverArm())
            # Gal longitude angle corresponding to chi:
            chi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Phi())
            # Gal longitude angle corresponding to chi:
            psi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Theta())
                
        # Initialize arrays:
        erg = np.array(erg)
        tt = np.array(tt)
        et = np.array(et)

        latX = np.array(latX)
        lonX = np.array(lonX)
        # Change longitudes to from 0..360 deg to -180..180 deg
        lonX[lonX > np.pi] -= 2*np.pi

        latZ = np.array(latZ)
        lonZ = np.array(lonZ)
        # Change longitudes to from 0..360 deg to -180..180 deg
        lonZ[lonZ > np.pi] -= 2*np.pi

        phi = np.array(phi)

        chi_loc = np.array(chi_loc)

        # Change azimuth angle to 0..360 deg
        chi_loc[chi_loc < 0] += 2*np.pi

        psi_loc = np.array(psi_loc)
    
        dist = np.array(dist)

        chi_gal = np.array(chi_gal)
        psi_gal = np.array(psi_gal)
        self.chi_gal_old = chi_gal
        self.psi_gal_old = psi_gal
        print()
        print("chi gal:")
        print("max: " + str(np.amax(chi_gal)))
        print("min: " + str(np.amin(chi_gal)))
        print(chi_gal)
        print()
        print("psi gal:")
        print("max: " + str(np.amax(psi_gal)))
        print("min: " + str(np.amin(psi_gal)))
        print(psi_gal)
        
        
        self.compare(self.psi_gal_old,self.chi_gal_new,"chi_gal")

        # Construct Y direction from X and Z direction
        lonlatY = self.construct_scy(np.rad2deg(lonX),np.rad2deg(latX),
                                np.rad2deg(lonZ),np.rad2deg(latZ))
        lonY = np.deg2rad(lonlatY[0])
        latY = np.deg2rad(lonlatY[1])

        # Avoid negative zeros
        chi_loc[np.where(chi_loc == 0.0)] = np.abs(chi_loc[np.where(chi_loc == 0.0)])
        
        # Make observation dictionary
        cosi_dataset = {'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings':np.array([lonX,latX]).T,
                        'Ypointings':np.array([lonY,latY]).T,
                        'Zpointings':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal} 
        self.cosi_dataset = cosi_dataset

        # Write unbinned data to file (either fits or hdf5):
        self.write_unbinned_output() 
        
        return 

