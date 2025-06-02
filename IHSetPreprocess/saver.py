import xarray as xr
from datetime import datetime
import numpy as np
import json
from .interpolator import interpolator
from scipy.stats import circmean, circstd
from pyproj import CRS, Transformer



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
        self.rot = None
        self.mask_nan_rot = None
        self.x_pivotal = None
        self.y_pivotal = None
        self.phi_pivotal = None
        self.average_obs = None
        self.mask_nan_average_obs = None

        
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
            self.lat_w = np.concatenate((self.lat_w, wave_data.lat), axis=0)
            self.lon_w = np.concatenate((self.lon_w, wave_data.lon), axis=0)
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

        self.lat_w = np.array(self.lat_w).flatten()
        self.lon_w = np.array(self.lon_w).flatten()

        if self.waves_epsg != 4326:
            crs_from = CRS.from_epsg(self.waves_epsg)
            crs_to = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(crs_from, crs_to)
            self.lon_w, self.lat_w = transformer.transform(self.lon_w, self.lat_w)

        # Calcular estatísticas para cada variável
        hs_attrs = {
            "units": "Meters",
            "standard_name": "wave_significant_height",
            "long_name": "Wave Significant Height",
            "max_value": np.nanmax(self.hs),
            "min_value": np.nanmin(self.hs),
            "mean_value": np.nanmean(self.hs),
            "standard_deviation": np.nanstd(self.hs)
        }
        tp_attrs = {
            "units": "Seconds",
            "standard_name": "wave_peak_period",
            "long_name": "Wave Peak Period",
            "max_value": np.nanmax(self.tp),
            "min_value": np.nanmin(self.tp),
            "mean_value": np.nanmean(self.tp),
            "standard_deviation": np.nanstd(self.tp)
        }
        dir_attrs = {
            "units": "Degrees North",
            "standard_name": "wave_direction",
            "long_name": "Wave Direction of Propagation",
            "max_value": np.nanmax(self.dir),
            "min_value": np.nanmin(self.dir),
            "circular_mean": circmean(self.dir, high=360, low=0, nan_policy='omit'),
            "circular_standard_deviation": circstd(self.dir, high=360, low=0, nan_policy='omit')
        }
        tide_attrs = {
            "units": "Meters",
            "standard_name": "astronomical_tide",
            "long_name": "Astronomical Tide",
            "max_value": np.nanmax(self.tide),
            "min_value": np.nanmin(self.tide),
            "mean_value": np.nanmean(self.tide),
            "standard_deviation": np.nanstd(self.tide)
        }
        surge_attrs = {
            "units": "Meters",
            "standard_name": "storm_surge",
            "long_name": "Storm Surge",
            "max_value": np.nanmax(self.surge),
            "min_value": np.nanmin(self.surge),
            "mean_value": np.nanmean(self.surge),
            "standard_deviation": np.nanstd(self.surge)
        }
        slr_attrs = {
            "units": "Meters",
            "standard_name": "sea_level_rise",
            "long_name": "Sea Level Rise",
            "max_value": np.nanmax(self.slr),
            "min_value": np.nanmin(self.slr),
            "mean_value": np.nanmean(self.slr),
            "standard_deviation": np.nanstd(self.slr)
        }
        obs_attrs = {
            "units": "Meters",
            "standard_name": "shoreline_position",
            "long_name": "Shoreline Position",
            "max_value": np.nanmax(self.obs),
            "min_value": np.nanmin(self.obs),
            "mean_value": np.nanmean(self.obs),
            "standard_deviation": np.nanstd(self.obs)
        }
        rot_attrs = {
            "units": "Degrees",
            "standard_name": "shoreline_rotation",
            "long_name": "Shoreline Rotation",
            "max_value": np.nanmax(self.rot),
            "min_value": np.nanmin(self.rot),
            "mean_value": circmean(self.rot),
            "standard_deviation": circstd(self.rot)
        }
        avg_obs_attrs = {
            "units": "Meters",
            "standard_name": "average_shoreline_position",
            "long_name": "Average Shoreline Position",
            "max_value": np.nanmax(self.average_obs),
            "min_value": np.nanmin(self.average_obs),
            "mean_value": np.nanmean(self.average_obs),
            "standard_deviation": np.nanstd(self.average_obs)
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
                "rot": ("time_obs", self.rot, rot_attrs),
                "average_obs": ("time_obs", self.average_obs, avg_obs_attrs),
                "mask_nan_obs": (("time_obs", "ntrs"), self.mask_nan_obs, {
                    "units": "Boolean",
                    "standard_name": "mask_nan_obs",
                    "long_name": "Mask for NaNs in observations"
                }),
                "mask_nan_rot": ("time_obs", self.mask_nan_rot, {
                    "units": "Boolean",
                    "standard_name": "mask_nan_rot",
                    "long_name": "Mask for NaNs in rotation"
                }),
                "mask_nan_average_obs": ("time_obs", self.mask_nan_average_obs, {
                    "units": "Boolean",
                    "standard_name": "mask_nan_average_obs",
                    "long_name": "Mask for NaNs in average observations"
                }),

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
                }),
                "x_pivotal": ("x_pivotal", self.x_pivotal, {
                    "units": "meters",
                    "standard_name": "x_pivotal",
                    "long_name": "Initial x coordinate of pivotal transect"
                }),
                "y_pivotal": ("y_pivotal", self.y_pivotal, {
                    "units": "meters",
                    "standard_name": "y_pivotal",
                    "long_name": "Initial y coordinate of pivotal transect"
                }),
                "phi_pivotal": ("phi_pivotal", self.phi_pivotal, {
                    "units": "degrees",
                    "standard_name": "phi_pivotal",
                    "long_name": "Angle of pivotal transect"
                }),

            },
            attrs=self.attrs
        )
        
        # Export to NetCDF
        ds.to_netcdf(filepath, engine="netcdf4")

        if self.obs_dataSource != 'CSV file (transects)':
            # Export transects to a CSV file
            transects_filepath = filepath.replace('.nc', '_transects.csv')
            export_transects(self.xi, self.yi, self.xf, self.yf, self.phi, self.epsg, transects_filepath)

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

        if self.ntrs > 1:
            self.rot, self.mask_nan_rot = calculate_rotation(self.xi, self.yi, self.phi, self.obs)
            pivotal = find_pivotal_point(self.obs, self.xi, self.yi, self.phi)
            if pivotal is not None:
                self.x_pivotal =[ pivotal['xi']]
                self.y_pivotal = [pivotal['yi']]
                self.phi_pivotal = [pivotal['phi']]
                # print(f"Pivotal point found at ({self.x_pivotal}, {self.y_pivotal}) with angle {self.phi_pivotal}")
                #now we find the closest transect to the pivotal point
                # dist = np.sqrt((self.xi - self.x_pivotal[0])**2 + (self.yi - self.y_pivotal[0])**2)
                # idx = np.argmin(dist)
                # self.pivotal_trs = idx
            else:
                self.x_pivotal = [None]
                self.y_pivotal = [None]
                self.phi_pivotal = [None]
        else:
            self.rot = np.zeros_like(self.obs)
            self.mask_nan_obs = interp.mask_nan_obs
            self.x_pivotal = [None]
            self.y_pivotal = [None]
            self.phi_pivotal = [None]



        self.average_obs, self.mask_nan_average_obs = calculate_obs_average(self.obs)        


        self.hs = interp.hs
        self.tp = interp.tp
        self.dir = interp.dir
        self.tide = interp.tide
        self.surge = interp.surge
        self.slr = interp.slr
        self.time = interp.time
        self.lat = interp.lat
        self.lon = interp.lon
        self.mask_nan_obs = interp.mask_nan_obs


def calculate_rotation(X0, Y0, phi, dist):
    """
    Calculate the shoreline rotation.
    """

    phi_rad = np.deg2rad(phi)

    mean_shoreline = np.nanmean(dist, axis=0)

    detrended_dist = np.zeros(dist.shape)

    for i in range(dist.shape[1]):
        detrended_dist[:, i] = dist[:, i] - mean_shoreline[i]

    # We will calculate the rotation only for the times where we at least 80% of the data

    nans_rot = np.sum(np.isnan(detrended_dist), axis=1) > 0.2 * dist.shape[1]
    
    alpha = np.zeros(dist.shape[0]) * np.nan
    
    for i in range(dist.shape[0]):
        if not nans_rot[i]:
            XN, YN = X0 + detrended_dist[i, :] * np.cos(phi_rad), Y0 + detrended_dist[i, :] * np.sin(phi_rad)
            ii_nan = np.isnan(XN) | np.isnan(YN)
            fitter = np.polyfit(XN[~ii_nan], YN[~ii_nan], 1)
            alpha[i] = 90 - np.rad2deg(np.arctan(fitter[0]))
            if alpha[i] < 0:
                alpha[i] += 360

    mask_nans = np.isnan(alpha)

    # mean_alpha_ori = circmean(alpha[~mask_nans], high=360, low=0)
    # if mean_alpha_ori<0:
    #     mean_alpha_ori += 360

    mean_phi = 90 - circmean(phi, high=360, low=0)
    if mean_phi<0:
        mean_phi += 360
    mean_alpha = circmean(alpha[~mask_nans], high=360, low=0) + 90
    if mean_alpha<0:
        mean_alpha += 360   
    mean_alpha_2 = circmean(alpha[~mask_nans], high=360, low=0) - 90
    if mean_alpha_2<0:
        mean_alpha_2 += 360

    # print(f"Mean alpha: {mean_alpha}, Mean phi: {mean_phi}, Mean alpha 2: {mean_alpha_2}, Mean Alpha_ori: {mean_alpha_ori}")

    if np.abs(mean_alpha - mean_phi) <= np.abs(mean_alpha_2 - mean_phi) :
        alpha  += 90
    else:
        alpha -= 90
    
    # Now we change the <0 to 0-360

    alpha[alpha < 0] += 360

    return alpha, mask_nans

from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
def find_pivotal_point(obs, xi, yi, phi):
    """
    Find the pivotal point of the shoreline
    """
    # Interpolação para lidar com valores NaN em obs
    
    Obs_interp = np.zeros_like(obs)
    for i in range(obs.shape[1]):
        Obs_interp[:, i] = np.interp(
            np.arange(len(obs)),
            np.flatnonzero(~np.isnan(obs[:, i])),
            obs[~np.isnan(obs[:, i]), i]
        )

    nans_rot = np.sum(np.isnan(obs), axis=1) > 0.1 * obs.shape[1]
    Obs_interp = Obs_interp[~nans_rot, :]

    
    # Aplicando PCA
    pca = PCA(n_components=obs.shape[1])
    pca.fit(Obs_interp)
    u = pca.components_.T

    # Calculando a distância ao longo da linha xi, yi
    d = np.sqrt((xi - xi[0]) ** 2 + (yi - yi[0]) ** 2)

    # Encontrando a interseção entre o 1º e 2º modos
    # Calculando a diferença entre os dois modos
    diff = u[:, 0] - u[:, 1]

    # Procurando as mudanças de sinal, que indicam pontos de interseção
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        print("Nenhuma interseção encontrada entre os dois primeiros modos.")
        return None

    # Pega o primeiro ponto de interseção encontrado
    idx = sign_changes[0]

    # Interpolação linear para encontrar a posição exata da interseção
    x1, x2 = d[idx], d[idx + 1]
    y1, y2 = diff[idx], diff[idx + 1]

    # Calculando o ponto de interseção usando interpolação linear
    d_intersect = x1 - y1 * (x2 - x1) / (y2 - y1)

    # Interpolando para encontrar as coordenadas xi, yi e o ângulo phi na interseção
    xi_intersect = np.interp(d_intersect, d, xi)
    yi_intersect = np.interp(d_intersect, d, yi)
    phi_intersect = np.interp(d_intersect, d, phi)

    pivotal_point = {
        'xi': xi_intersect,
        'yi': yi_intersect,
        'phi': phi_intersect
    }

    # # Plot para visualizar os modos e o ponto de interseção
    # plt.figure()
    # plt.plot(d, u[:, 0], color=[0.4, 0.4, 0.4], linewidth=2, label="1st mode")
    # plt.plot(d, u[:, 1], 'k', linewidth=3, label="2nd mode")
    # plt.axvline(d_intersect, color='r', linestyle='--', label='Intersection')
    # plt.grid(True)
    # plt.ylabel("e_n(y)")
    # plt.xlabel("Alongshore distance (m)")
    # plt.legend()
    # plt.show()

    return pivotal_point

def calculate_obs_average(obs):
    """
    Calculate the average of the observations
    """

    nans_rot = np.sum(np.isnan(obs), axis=1) > 0.2 * obs.shape[1]

    mean_obs = np.nanmean(obs, axis=1)

    mean_obs[nans_rot] = np.nan

    mask_nan = np.isnan(mean_obs)

    return mean_obs, mask_nan

def export_transects(xi, yi, xf, yf, phi, epsg, filepath):
    """
    Export the transects to a .csv file
    """
    import pandas as pd

    data = {
        'xi': xi,
        'yi': yi,
        'xf': xf,
        'yf': yf,
        'phi': phi,
        'epsg': epsg
    }

    df = pd.DataFrame(data)

    df.to_csv(filepath, index=False)

    print(f"Transects exported to {filepath}")
