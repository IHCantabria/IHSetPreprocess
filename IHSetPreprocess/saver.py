import xarray as xr
from datetime import datetime
import numpy as np

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
        
    def add_waves(self, wave_data):
        """ Add wave data to the dataset """

        if self.hs is None:
            self.hs = wave_data.hs
            self.tp = wave_data.tp
            self.dir = wave_data.dir
            self.lat = wave_data.lat
            self.lon = wave_data.lon
            self.time = wave_data.time
            self.w_dataSource = wave_data.dataSource
        else:
            self.hs = np.concatenate((self.hs, wave_data.hs), axis=1)
            self.tp = np.concatenate((self.tp, wave_data.tp), axis=1)
            self.dir = np.concatenate((self.dir, wave_data.dir), axis=1)
            self.lat = np.concatenate((self.lat, wave_data.lat), axis=0)
            self.lon = np.concatenate((self.lon, wave_data.lon), axis=0)
            self.w_dataSource = self.w_dataSource+'/'+wave_data.dataSource

        
    def add_sl(self, sl_data):
        """ Add sea level data to the dataset """
        self.sl = sl_data
        
    def add_obs(self, obs_data):
        """ Add observation data to the dataset """
        self.obs = obs_data

    def set_attrs(self, **kwargs):
        """ Set global attributes """
        # Global attributes

        data_sources = f'Waves: {self.w_dataSource}, Surge: {self.sl.dataSource_surge}, Tide: {self.sl.dataSource_tide}, Obs: {self.obs.dataSource}'
        
        creation_date = datetime.now().strftime("%Y-%m-%d")

        self.attrs = {
            "title": "Input File for IH-SET models",
            "institution": "Environmental Hydraulics Institute of Cantabria - https://ihcantabria.com/",
            "source": "IH-SET preprocessing module",
            "history": f'Created on {creation_date}.',
            "references": "Jaramillo et al. (2025) - doi: xxxxxxxx.xx",
            "Documentation": "https://ihcantabria.github.io/IHSetDocs/",
            "Conventions": "CF-1.6",
            "Data Sources": data_sources,
            "summary": "This dataset is output from the IH-SET preprocessing module. Etc…",
            "geospatial_lat_min": -90,
            "geospatial_lat_max": 90,
            "geospatial_lon_min": -180,
            "geospatial_lon_max": 180
        }

    def export_netcdf(self, filepath, **kwargs):
        """ Export the dataset to a NetCDF file using xarray """
        # Create dataset with xarray
        ds = xr.Dataset(
            {
                "hs": (("time", "ntrs"), self.hs, {
                    "units": "Meters",
                    "standard_name": "wave_significant_height",
                    "long_name": "Wave Significant Height"
                }),
                "tp": (("time", "ntrs"), self.tp, {
                    "units": "Seconds",
                    "standard_name": "wave_peak_period",
                    "long_name": "Wave Peak Period"
                }),
                "dir": (("time", "ntrs"), self.dir, {
                    "units": "Degrees North",
                    "standard_name": "wave_direction",
                    "long_name": "Wave Direction of Propagation"
                }),
                "tide": (("time", "ntrs"), self.sl.tide, {
                    "units": "Meters",
                    "standard_name": "astronomical_tide",
                    "long_name": "Astronomical Tide"
                }),
                "surge": (("time", "ntrs"), self.sl.surge, {
                    "units": "Meters",
                    "standard_name": "storm_surge",
                    "long_name": "Storm Surge"
                }),
                # "slr": ("time", self.sl.slr, {
                #     "units": "Meters",
                #     "standard_name": "sea_level_rise",
                #     "long_name": "Sea Level Rise"
                # })
            },
            coords={
                "time": ("time", self.time, {
                    "standard_name": "time",
                    "long_name": "Time"
                }),
                "lat": ("lat", self.lat, {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "long_name": "Latitude"
                }),
                "lon": ("lon", self.lon, {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "long_name": "Longitude"
                }),
                "ntrs": ("ntrs", self.obs.ntrs, {
                    "units": "number_of",
                    "standard_name": "number_of_trs",
                    "long_name": "Number of Transects"
                })
            },
            attrs=self.attrs
        )
        
        # Export to NetCDF
        ds.to_netcdf(filepath, engine="netcdf4")
        print(f"File {filepath} saved correctly.")


