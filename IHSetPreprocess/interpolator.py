# Now we need to check the time and space consistency of the data. We will use the check_times and check_dimensions methods to do this.
import numpy as np

class interpolator(object):
    """
    interpolator
    
    Interpolating the observation data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """

    def __init__(self, waves, sl, obs):
        """
        Define the file path, data source, and dataset
        """

        self.waves = waves
        self.sl = sl
        self.obs = obs
        self.time = None
        self.lat = None
        self.lon = None
    
    def check_times(self):

        """
        Check the time consistency of the data
        """
        
        if self.waves.time == self.sl.time_surge == self.sl.time_tide:
            self.time = self.waves.time
            return 'Time is consistent'
        else:
            self.interp_time()
            return 'Interpolating time'
    
    def check_dimensions(self):
        """
        Check the space consistency of the data
        """
        
        if self.waves.lat == self.sl.lat == self.obs.lat and self.waves.lon == self.sl.lon == self.obs.lon:
            self.lat = self.waves.lat
            self.lon = self.waves.lon
            return 'Space is consistent'
        else:
            self.interp_space()
    
    def interp_time(self):
        """
        Interpolate the time dimension
        we need to interp sl variables into waves time
        """
        # Interpolate the time dimension
        self.time = self.waves.time
        
        if self.sl.surge is not None:
            self.sl.surge = np.interp(self.time, self.sl.time_surge, self.sl.surge)
        else:
            self.sl.surge = None
        
        if self.sl.tide is not None:
            self.sl.tide = np.interp(self.time, self.sl.time_tide, self.sl.tide)
        else:
            self.sl.tide = None
        
        if self.sl.slr is not None:
            self.sl.slr = np.interp(self.time, self.sl.time_slr, self.sl.slr)
        else:
            self.sl.slr = None
        
        self.sl.time_surge = self.time
        self.sl.time_tide = self.time
    
    def interp_space(self):
        """
        Interpolate the space dimension
        we need to interp sl variables into waves space
        """
        # Interpolate the space dimension
        self.lat = self.waves.lat
        self.lon = self.waves.lon
        lat_sl = self.sl.lat
        lon_sl = self.sl.lon

        if self.sl.surge is not None:
            self.sl.surge = np.interp(self.lat, lat_sl, self.sl.surge)
            self.sl.surge = np.interp(self.lon, lon_sl, self.sl.surge)
        else:
            self.sl.surge = None
        
        if self.sl.tide is not None:
            self.sl.tide = np.interp(self.lat, lat_sl, self.sl.tide)
            self.sl.tide = np.interp(self.lon, lon_sl, self.sl.tide)
        else:
            self.sl.tide = None
        
        if self.sl.slr is not None:
            self.sl.slr = np.interp(self.lat, lat_sl, self.sl.slr)
            self.sl.slr = np.interp(self.lon, lon_sl, self.sl.slr)
        else:
            self.sl.slr = None
        
        self.sl.lat = self.lat
        self.sl.lon
   

            
            



