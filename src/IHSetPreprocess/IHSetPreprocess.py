import scipy.io
import pandas as pd
import xarray as xr

class pre_config(object):
    """
    pre_config
    
    Preprocessing the configuration for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path, cfg):
            self.path = path
            config = xr.Dataset(coords=cfg)
            config.to_netcdf(self.path+'/config.nc', engine='netcdf4')
            config.close()

class pre_wav_csv(object):
    """
    pre_wav_csv
    
    Preprocessing the wave data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path, path_Hs, path_Tp, path_Dir):
        """
        Wave csv files (Copernicus, buoy, etc.)
        
        Each csv files for Hs, Tp, Dir should contain the followings:
        
        Datetime : YYYY-MM-DD (Shoreshop) or YYYY-MM-DDThh:mm:ss.sssZ(Copernicus)
        
        Variable : Hs, Tp, Dir
        
        """
        Hs = pd.read_csv(path_Hs)
        Tp = pd.read_csv(path_Tp)
        Dir = pd.read_csv(path_Dir)
        
        Time = Hs['Datetime'].values
        dt = pd.to_datetime(Time)

        wav = xr.Dataset(coords={
                'Y': dt.year.values,
                'M': dt.month.values,
                'D': dt.day.values,
                'h': dt.hour.values,
                'Hs': Hs['Hs'].values,
                'Tp': Tp['Tp'].values,
                'Dir': Dir['Dir'].values
                })
        
        wav.to_netcdf(path+'/wav.nc', engine='netcdf4')
        
class pre_sl_csv(object):
    """
    pre_sl_csv
    
    Preprocessing the sealevel data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path, path_tide, path_surge, path_sl):
        """
        Wave csv files (Copernicus, buoy, etc.)
        
        Each csv files for Hs, Tp, Dir should contain the followings:
        
        Datetime : YYYY-MM-DD (Shoreshop) or YYYY-MM-DDThh:mm:ss.sssZ(Copernicus)
        
        Variable : Tide, Surge, Sealevel
        
        """
        Tide = pd.read_csv(path_tide)
        Surge = pd.read_csv(path_surge)
        Sealevel = pd.read_csv(path_sl)
        
        Time = Tide['Datetime'].values
        dt = pd.to_datetime(Time)

        sl = xr.Dataset(coords={
                'Y': dt.year.values,
                'M': dt.month.values,
                'D': dt.day.values,
                'h': dt.hour.values,
                'tide': Tide['Tide'].values,
                'surge': Surge['Surge'].values,
                'sl': Sealevel['Sealevel'].values
                })

        sl.to_netcdf(path+'/sl.nc', engine='netcdf4')
        
class pre_wav_ih(object):
    """
    pre_wav_ih
    
    Preprocessing the wave data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path, path_GOW):
        """
        Wave IH-Data files
                
        GOW file in IH-Data should contain the followings:
        
        Datetime : ['time'] datenum for matlab
        
        Variable : ['hs'] for Hs, ['t02'] for Tp, ['dir'] for Dir
        
        """
        IH_Data_GOW = scipy.io.loadmat(path_GOW)
        Time = IH_Data_GOW["data"]['time'][0][0].flatten()
        dates = pd.to_datetime(Time - 719529, unit='d').round('s').to_pydatetime()

        wav = xr.Dataset(coords={
                'Y': ('time', [dt.year for dt in dates]),
                'M': ('time', [dt.month for dt in dates]),
                'D': ('time', [dt.day for dt in dates]),
                'h': ('time', [dt.hour for dt in dates]),
                'Hs': IH_Data_GOW["data"]['hs'][0][0].flatten(),
                'Tp': IH_Data_GOW["data"]['t02'][0][0].flatten(),
                'Dir': IH_Data_GOW["data"]['dir'][0][0].flatten()
                })

        wav.to_netcdf(path+'/wav.nc', engine='netcdf4')
        
class pre_sl_ih(object):
    """
    pre_sl_ih
    
    Preprocessing the sea level data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path, path_GOS, path_GOT):
        """
        Wave IH-Data files
                
        GOT and GOW files in IH-Data should contain the followings:
        
        Datetime : ['time'] datenum for matlab
        
        Variable : ['hs'] for Hs, ['t02'] for Tp, ['dir'] for Dir
        
        """
        IH_Data_GOS = scipy.io.loadmat(path_GOS)
        IH_Data_GOT = scipy.io.loadmat(path_GOT)
        Time = IH_Data_GOS["data"]['time'][0][0].flatten()
        dates = pd.to_datetime(Time - 719529,unit='d').round('s').to_pydatetime()

        sl = xr.Dataset(coords={
                'Y': ('time', [dt.year for dt in dates]),
                'M': ('time', [dt.month for dt in dates]),
                'D': ('time', [dt.day for dt in dates]),
                'h': ('time', [dt.hour for dt in dates]),
                'surge': IH_Data_GOS["data"]['zeta'][0][0].flatten(),
                'tide': IH_Data_GOT["data"]['tide'][0][0].flatten()
                })
        
        sl.to_netcdf(path+'/sl.nc', engine='netcdf4')
        
class pre_obs(object):    
    """
    pre_obs
    
    Preprocessing the observation data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    
    """
    def __init__(self, path, path_obs):
            
        """
        Shoreline or Orientation observation files (Shoreshop, GPS)
        
        Each csv files for observation should contain the followings:
        
        Datetime : YYYY-MM-DD (Shoreshop)
        
        Variable : Observation (Shoreline or Orientation)
        
        """
        Obs = pd.read_csv(path_obs)

        Time = Obs['Datetime'].values
        dt = pd.to_datetime(Time)

        ens = xr.Dataset(coords={
                'Y': dt.year.values,
                'M': dt.month.values,
                'D': dt.day.values,
                'h': dt.hour.values,
                'Obs': Obs['Observation'].values
                })

        ens.to_netcdf(path+'/ens.nc', engine='netcdf4')