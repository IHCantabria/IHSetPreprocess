import xarray as xr
from datetime import datetime
import numpy as np
import json
from .interpolator import interpolator
from scipy.stats import circmean, circstd

class save_SET_standard_netCDF(object):
    """
    save_SET_standard_netCDF
    
    Save the preprocessed data in the standard format for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self):
        # Dimensions
        self.lat = None
        self.lon = None
        self.time = None
        self.ntrs = None
        self.lon_w = None
        self.lat_w = None
        
        # Variables
        self.w_dataSource = None
        self.sl_dataSource_surge = None
        self.sl_dataSource_tide = None
        self.obs_dataSource = None
        self.hs = None
        self.tp = None
        self.dir = None
        self.tide = None
        self.surge = None
        self.slr = None
        self.obs = None
        self.time_tide = None
        self.time_surge = None
        self.time_obs = None
        self.time_slr = None
        self.xi = None
        self.yi = None
        self.xf = None
        self.yf = None
        self.phi = None
        self.epsg = None
        self.attrs = None
        self.applicable_models = None
        self.waves_epsg = None
        self.mask_nan_obs = None

        
    def add_waves(self, wave_data):
        """ Add wave data to the dataset """

        if self.hs is None:
            self.hs = wave_data.hs
            self.tp = wave_data.tp
            self.dir = wave_data.dir
            self.lat_w = wave_data.lat
            self.lon_w = wave_data.lon
            self.time = wave_data.time
            self.w_dataSource = wave_data.dataSource
            self.waves_epsg = wave_data.epsg
        else:
            self.hs = np.concatenate((self.hs, wave_data.hs), axis=1)
            self.tp = np.concatenate((self.tp, wave_data.tp), axis=1)
            self.dir = np.concatenate((self.dir, wave_data.dir), axis=1)
            self.lat = np.concatenate((self.lat, wave_data.lat), axis=0)
            self.lon = np.concatenate((self.lon, wave_data.lon), axis=0)
            self.w_dataSource = self.w_dataSource+'/'+wave_data.dataSource

        
    def add_sl(self, sl_data):
        """ Add sea level data to the dataset """

        if self.tide is None:
            self.tide = sl_data.tide
            self.surge = sl_data.surge
            self.slr = sl_data.slr
            self.time_surge = sl_data.time_surge
            self.time_tide = sl_data.time_tide
            self.time_slr = sl_data.time_slr
            self.dataSource_surge = sl_data.dataSource_surge
            self.dataSource_tide = sl_data.dataSource_tide
        else:
            self.tide = np.concatenate((self.tide, sl_data.tide), axis=1)
            self.surge = np.concatenate((self.surge, sl_data.surge), axis=1)
            self.slr = np.concatenate((self.slr, sl_data.slr), axis=1)
            self.dataSource_surge = self.sl_dataSource_surge+'/'+sl_data.dataSource_surge
            self.dataSource_tide = self.sl_dataSource_tide+'/'+sl_data.dataSource_tide
        
    def add_obs(self, obs_data):
        """ Add observation data to the dataset """
       
        self.obs = obs_data.obs
        self.time_obs = obs_data.time_obs
        self.ntrs = obs_data.ntrs
        self.obs_dataSource = obs_data.dataSource
        self.xi = obs_data.xi
        self.yi = obs_data.yi
        self.xf = obs_data.xf
        self.yf = obs_data.yf
        self.phi = obs_data.phi
        self.epsg = obs_data.epsg

    def set_attrs(self):
        """ Set global attributes """
        # Global attributes

        data_sources = f'Waves: {self.w_dataSource}, Surge: {self.dataSource_surge}, Tide: {self.dataSource_tide}, Obs: {self.obs_dataSource}'

        # Lets use the date with YYY-MM-DDThh:mm:ssZ format        
        creation_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        self.check_models()

        self.attrs = {
            "title": "Input File for IH-SET models",
            "institution": "Environmental Hydraulics Institute of Cantabria - https://ihcantabria.com/",
            "source": "IH-SET preprocessing module",
            "history": f'Created on {creation_date}',
            "references": "Jaramillo et al. (2025) - doi: xxxxxxxx.xx",
            "Documentation": "https://ihcantabria.github.io/IHSetDocs/",
            "Conventions": "CF-1.6",
            "Data Sources": data_sources,
            "summary": "This dataset is output from the IH-SET preprocessing module. Etc…",
            "geospatial_lat_min": -90,
            "geospatial_lat_max": 90,
            "geospatial_lon_min": -180,
            "geospatial_lon_max": 180,
            "applicable_models": self.applicable_models,
            "EPSG": self.epsg,
        }

    def export_netcdf(self, filepath):
        """ Export the dataset to a NetCDF file using xarray """
        # Calcular estatísticas para cada variável
        hs_attrs = {
            "units": "Meters",
            "standard_name": "wave_significant_height",
            "long_name": "Wave Significant Height",
            "max_value": np.nanmax(self.hs),
            "min_value": np.nanmin(self.hs),
            "standard_deviation": np.nanstd(self.hs)
        }
        tp_attrs = {
            "units": "Seconds",
            "standard_name": "wave_peak_period",
            "long_name": "Wave Peak Period",
            "max_value": np.nanmax(self.tp),
            "min_value": np.nanmin(self.tp),
            "standard_deviation": np.nanstd(self.tp)
        }
        dir_attrs = {
            "units": "Degrees North",
            "standard_name": "wave_direction",
            "long_name": "Wave Direction of Propagation",
            "circular_mean": circmean(self.dir, high=360, low=0, nan_policy='omit'),
            "circular_standard_deviation": circstd(self.dir, high=360, low=0, nan_policy='omit')
        }
        tide_attrs = {
            "units": "Meters",
            "standard_name": "astronomical_tide",
            "long_name": "Astronomical Tide",
            "max_value": np.nanmax(self.tide),
            "min_value": np.nanmin(self.tide),
            "standard_deviation": np.nanstd(self.tide)
        }
        surge_attrs = {
            "units": "Meters",
            "standard_name": "storm_surge",
            "long_name": "Storm Surge",
            "max_value": np.nanmax(self.surge),
            "min_value": np.nanmin(self.surge),
            "standard_deviation": np.nanstd(self.surge)
        }
        slr_attrs = {
            "units": "Meters",
            "standard_name": "sea_level_rise",
            "long_name": "Sea Level Rise",
            "max_value": np.nanmax(self.slr),
            "min_value": np.nanmin(self.slr),
            "standard_deviation": np.nanstd(self.slr)
        }
        obs_attrs = {
            "units": "Meters",
            "standard_name": "shoreline_position",
            "long_name": "Shoreline Position",
            "max_value": np.nanmax(self.obs),
            "min_value": np.nanmin(self.obs),
            "standard_deviation": np.nanstd(self.obs)
        }

        # Create dataset with xarray
        ds = xr.Dataset(
            {
                "hs": (("time", "ntrs"), self.hs, hs_attrs),
                "tp": (("time", "ntrs"), self.tp, tp_attrs),
                "dir": (("time", "ntrs"), self.dir, dir_attrs),
                "tide": (("time", "ntrs"), self.tide, tide_attrs),
                "surge": (("time", "ntrs"), self.surge, surge_attrs),
                "slr": (("time", "ntrs"), self.slr, slr_attrs),
                "obs": (("time_obs", "ntrs"), self.obs, obs_attrs),
                "mask_nan_obs": (("time_obs", "ntrs"), self.mask_nan_obs, {
                    "units": "Boolean",
                    "standard_name": "mask_nan_obs",
                    "long_name": "Mask for NaNs in observations"
                })
            },
            coords={
                "time": ("time", self.time, {
                    "standard_name": "time",
                    "long_name": "Time"
                }),
                "lat": ("lat", self.lat, {
                    "units": "degrees_north",
                    "standard_name": "latitude_of_waves",
                    "long_name": "Latitude of waves"
                }),
                "lon": ("lon", self.lon, {
                    "units": "degrees_east",
                    "standard_name": "longitude_of_waves",
                    "long_name": "Longitude of waves"
                }),
                "ntrs": ("ntrs", np.arange(self.ntrs), {
                    "units": "number_of",
                    "standard_name": "number_of_trs",
                    "long_name": "Number of Transects"
                }),
                "time_obs": ("time_obs", self.time_obs, {
                    "standard_name": "time_of_observations",
                    "long_name": "Time of Observation"
                }),
                "xi": ("xi", self.xi, {
                    "units": "meters",
                    "standard_name": "xi_coordinate",
                    "long_name": "Origin x coordinate of transect"
                }),
                "yi": ("yi", self.yi, {
                    "units": "meters",
                    "standard_name": "yi_coordinate",
                    "long_name": "Origin y coordinate of transect"
                }),
                "xf": ("xf", self.xf, {
                    "units": "meters",
                    "standard_name": "xf_coordinate",
                    "long_name": "End x coordinate of transect"
                }),
                "yf": ("yf", self.yf, {
                    "units": "meters",
                    "standard_name": "yf_coordinate",
                    "long_name": "End y coordinate of transect"
                }),
                "phi": ("phi", self.phi, {
                    "units": "degrees",
                    "standard_name": "transect_angle",
                    "long_name": "Cartesian angle of transect"
                }),
                "lat_waves": ("lat_waves", self.lat_w, {
                    "units": "degrees_north",
                    "standard_name": "latitude_waves",
                    "long_name": "Latitude of provided waves"
                }),
                "lon_waves": ("lon_waves", self.lon_w, {
                    "units": "degrees_east",
                    "standard_name": "longitude_waves",
                    "long_name": "Longitude of provided waves"
                })
            },
            attrs=self.attrs
        )
        
        # Export to NetCDF
        ds.to_netcdf(filepath, engine="netcdf4")

        print(f"File {filepath} saved correctly.")

    def check_models(self):
        """
        Check wich model is applicable with the provided data
        """
        self.check_consistency()

        models = {'M&D':False,
                  'Y09':False, 
                  'SF':False, 
                  'JA20':False, 
                  'Lim':False, 
                  'Jara':False, 
                  'Turki':False, 
                  'JA21':False, 
                  'H&K':False,
                  'MOOSE':False}
        
        if np.sum(self.hs) != 0:
            models['Y09'] = True
            models['JA20'] = True
            models['Lim'] = True
            models['Jara'] = True
        
        if np.sum(self.hs) != 0 and np.sum(self.tp) != 0:
            models['SF'] = True

        if self.hs is not None and self.tp is not None and self.dir is not None:
            models['JA21'] = True
            models['H&K'] = True
            models['MOOSE'] = True
            models['M&D'] = True
            models['Turki'] = True
        
        models_json = json.dumps(models)
        self.applicable_models = models_json
    
    def check_consistency(self):
        """
        Check the consistency of the data
        """
        interp = interpolator(
            self.hs, self.tp, self.dir, self.tide, self.surge, self.slr, self.obs, self.time, self.time_surge, self.time_tide, self.time_slr,self.lat_w, self.lon_w, self.xf, self.yf, self.epsg, self.waves_epsg
        )

        interp.check_times()
        interp. check_dimensions()

        self.hs = interp.hs
        self.tp = interp.tp
        self.dir = interp.dir
        self.tide = interp.tide
        self.surge = interp.surge
        self.slr = interp.slr
        self.obs = interp.obs
        self.time = interp.time
        self.lat = interp.lat
        self.lon = interp.lon
        self.mask_nan_obs = interp.mask_nan_obs

        



