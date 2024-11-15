# Now we need to check the time and space consistency of the data. We will use the check_times and check_dimensions methods to do this.
import numpy as np
from pyproj import CRS, Transformer
import pandas as pd

class interpolator(object):
    """
    interpolator
    
    Interpolating the observation data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """

    def __init__(self, hs, tp, dir, tide, surge, slr, obs, time, time_surge, time_tide, time_slr, lat, lon, xf, yf, epsg, waves_epsg):
        """
        Define the file path, data source, and dataset
        """

        self.hs = hs
        self.tp = tp
        self.dir = dir
        self.tide = tide
        self.surge = surge
        self.slr = slr
        self.obs = obs
        self.time = time
        self.time_surge = time_surge
        self.time_tide = time_tide
        self.time_slr = time_slr
        self.lat = lat
        self.lon = lon
        self.xf = xf
        self.yf = yf
        self.epsg = epsg
        if waves_epsg is None:
            self.waves_epsg = 4326
        else:
            self.waves_epsg = waves_epsg
        
        
        self.erase_nones()
        self.make_mask_nan_obs()
        self.fill_nans()

        
    
    def check_times(self):

        """
        Check the time consistency of the data
        """
        if len(self.time) == len(self.time_surge) == len(self.time_tide):
            if np.sum(self.time == self.time_surge == self.time_tide):
                return 'Time is consistent'
            else:
                self.interp_time()
                return 'Interpolating time'
        else:
            self.interp_time()
            return 'Interpolating time'
    
    def check_dimensions(self):
        """
        Check the space consistency of the data
        """
        try:
            if self.lat == self.yf and self.lon == self.xf:
                self.interp_space()
                return 'Space is consistent'
            else:
                self.interp_space()
        except:
            print('Interpolating space')
            self.interp_space()
        
    def interp_time(self):
        """
        Interpolate the time dimension
        we need to interp sl variables into waves time
        """
        # Interpolate the time dimension

        timerr = np.vectorize(lambda x: pd.Timestamp(x).timestamp())

        t_float = timerr(self.time)
        t_float_s = timerr(self.time_surge)
        aux = np.zeros_like(self.hs)

        # Preencher os valores fora dos extremos com o valor médio das séries
        for i in range(np.shape(self.surge)[1]):
            mean_value = np.nanmean(self.surge[:, i])  # Calcula a média ignorando NaNs
            aux[:, i] = np.interp(t_float, t_float_s, self.surge[:, i], left=mean_value, right=mean_value)
        self.surge = aux

        t_float_t = timerr(self.time_tide)
        aux = np.zeros_like(self.hs)
        for i in range(np.shape(self.tide)[1]):
            mean_value = np.nanmean(self.tide[:, i])
            aux[:, i] = np.interp(t_float, t_float_t, self.tide[:, i], left=mean_value, right=mean_value)
        self.tide = aux

        t_float_slr = timerr(self.time_slr)
        aux = np.zeros_like(self.hs)
        for i in range(np.shape(self.slr)[1]):
            mean_value = np.nanmean(self.slr[:, i])
            aux[:, i] = np.interp(t_float, t_float_slr, self.slr[:, i], left=mean_value, right=mean_value)
        self.slr = aux
        
        self.time_surge = self.time
        self.time_tide = self.time
        self.time_slr = self.time

    def interp_space(self):
        """
        Interpolate the space dimension
        we need to interp sl variables into waves space
        """
        # Transform the waves coordinates
        if self.epsg != self.waves_epsg:
            crs = CRS.from_epsg(self.epsg)
            crs_waves = CRS.from_epsg(self.waves_epsg)
            transformer_waves_to_trs = Transformer.from_crs(crs_waves, crs)
            x_waves, y_waves = transformer_waves_to_trs.transform(self.lat, self.lon)
            x_waves = np.array(x_waves)
            y_waves = np.array(y_waves)
        else:
            x_waves = self.lat
            y_waves = self.lon
        
        crs_proj = CRS.from_epsg(self.epsg)
        crs_geo = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_proj, crs_geo)
        
        # Now we interpolate the variables to each transect endpoint (xf, yf)
        self.hs = interpWaves(self.xf, self.yf,x_waves, y_waves,  self.hs)
        self.tp = interpWaves(self.xf, self.yf,x_waves, y_waves,  self.tp)
        self.dir = interpWaves(self.xf, self.yf,x_waves, y_waves,  self.dir)
        self.tide = interpWaves(self.xf, self.yf, x_waves, y_waves, self.tide)
        self.surge = interpWaves(self.xf, self.yf, x_waves, y_waves, self.surge)
        self.slr = interpWaves(self.xf, self.yf, x_waves, y_waves, self.slr)
        self.lat = transformer.transform(self.xf, self.yf)[0]
        self.lon = transformer.transform(self.xf, self.yf)[1]
        
    def erase_nones(self):
        """
        Erase None values from the dataset
        """
        if self.surge is None:
            self.surge = np.zeros_like(self.hs)
            self.time_surge = self.time
        if self.tide is None:
            self.tide = np.zeros_like(self.hs)
            self.time_tide = self.time
        if self.slr is None:
            self.slr = np.zeros_like(self.hs)
            self.time_slr = self.time
        if self.dir is None:
            self.dir = np.zeros_like(self.hs)
        if self.tp is None:
            self.tp = np.zeros_like(self.hs)
    
    def make_mask_nan_obs(self):
        """
        Mask NaN values from the observations
        """
        self.mask_nan_obs = np.isnan(self.obs)
    
    def fill_nans(self):
        """
        Fill NaN values from the data interpolating the values
        """
        self.hs = fill_nan(self.hs)
        self.tp = fill_nan(self.tp)
        self.dir = fill_nan(self.dir)
        self.tide = fill_nan(self.tide)
        self.surge = fill_nan(self.surge)
        self.slr = fill_nan(self.slr)
        

def fill_nan(var):
    """
    Fill NaN values from the data interpolating the values
    """
    res = np.zeros_like(var)

    for i in range(var.shape[1]):
        if np.isnan(var[:,i]).any():
            mask = np.isnan(var[:,i])
            res[mask,i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), var[~mask,i])
            res[~mask,i] = var[~mask,i]
        else:
            res[:,i] = var[:,i]
    
    return res
   
from numba import jit

@jit
def interpWaves(x, y, xw, yw, var):

    d = np.sqrt((x-x[0]) ** 2 + (y-y[0]) ** 2)
    dd = np.sqrt((x[0]-xw) ** 2 + (y[0]-yw) ** 2)

    res = np.zeros((var.shape[0], len(x)))

    for i in range(var.shape[0]):
        res[i,:] =  np.interp(d, dd, var[i,:])

    return res
            



