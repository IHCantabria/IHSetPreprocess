import scipy.io
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import seaborn as sns
import numpy as np

class wave_data(object):
    """
    wave_data
    
    Preprocessing the wave data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path):
        """
        Define the file path, data source, and dataset
        """
        self.filePath = path
        self.dataSource = None
        self.data = None
        self.dataHs = None
        self.dataTp = None
        self.dataDir = None
        self.dataTime = None
        self.dt = None
        self.wav = None
        
        self.readWaves()
        if self.dt is not None:
                self.saveData(path)
                self.HsRose(path)
                self.TpRose(path)
                self.densityHsTp(path)
                
    def readWaves(self):
        """
        Read wave data
        """
        try:
                self.data = scipy.io.loadmat(self.filePath+'/GOW.mat')
                self.dataTime = self.data["data"]['time'][0][0].flatten()
                dates = pd.to_datetime(self.dataTime - 719529, unit='d').round('s').to_pydatetime()
                self.dt = pd.DatetimeIndex(dates)

                self.dataHs = self.data["data"]['hs'][0][0].flatten()
                self.dataTp = self.data["data"]['t02'][0][0].flatten()
                self.dataDir = self.data["data"]['dir'][0][0].flatten()
                self.dataSource = 'IH-DATA'
        except:
            pass

        try:
                Hs = pd.read_csv(self.filePath+'/Hs.csv')
                Tp = pd.read_csv(self.filePath+'/Tp.csv')
                Dir = pd.read_csv(self.filePath+'/Dir.csv')
                self.dataTime = Hs['Datetime'].values
                self.dt = pd.to_datetime(self.dataTime)
                
                self.dataHs = Hs['Hs'].values
                self.dataTp = Tp['Tp'].values
                self.dataDir = Dir['Dir'].values
                self.dataSource = 'CSV file'
        except:
            pass
                
        return 'Data loaded correctly'

    def saveData(self, path):
        """
        Save wave data
        """
        
        self.wav = xr.Dataset(coords={
                'Y': self.dt.year.values,
                'M': self.dt.month.values,
                'D': self.dt.day.values,
                'h': self.dt.hour.values,
                'Hs': self.dataHs,
                'Tp': self.dataTp,
                'Dir': self.dataDir
                })
        self.wav.to_netcdf(path+'/wav.nc', engine='netcdf4')
            
    def HsRose(self):
        """
        Plotting HsRose
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig = plt.figure(figsize=(6.5, 5), dpi=300, linewidth=5, edgecolor="#04253a")
        ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
        fig.add_axes(ax)
        
        data = pd.DataFrame({'X': self.wav['Dir'], 'Y': self.wav['Hs']})
        data = data.dropna()
        ax.bar(data['X'], data['Y'], normed=True, bins=[0, 0.5, 1, 1.5, 2, 2.5], opening=0.8, edgecolor='white')
        ax.set_legend(title=r"$Hs \, (m)$", loc='best')
        
        fig.savefig('./results/Rose_Wave'+'.png',dpi=300)
        
    def TpRose(self):
        """
        Plotting TpRose
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig = plt.figure(figsize=(6.5, 5), dpi=300, linewidth=5, edgecolor="#04253a")
        ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
        fig.add_axes(ax)
        
        data = pd.DataFrame({'X': self.wav['Dir'], 'Y': self.wav['Tp']})
        data = data.dropna()
        ax.bar(data['X'], data['Y'], normed=True, bins=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0], opening=0.8, edgecolor='white')
        ax.set_legend(title=r"$Tp \, (sec)$", loc='best')
        
        fig.savefig('./results/Rose_Wave'+'.png',dpi=300)
        
    def densityHsTp(self):
        """
        Plotting densityHsTp
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})
        font = {'family': 'serif',
                'weight': 'bold',
                'size': 8}

        XX = 'Hs'
        YY = 'Tp'
        data = pd.DataFrame({'X': self.wav[XX], 'Y': self.wav[YY]})
        data = data.dropna()

        fig = plt.figure(figsize=(5, 5), dpi=300, linewidth=5, edgecolor="#04253a")

        sns.kdeplot(data=data, x='X', y='Y', cmap='Blues', fill=True, thresh=0, levels=20)

        plt.xlim([np.floor(np.min(data['X'])),np.floor(np.max(data['X']))])
        plt.ylim([np.floor(np.min(data['Y'])),np.floor(np.max(data['Y']))])
        plt.xlabel(XX, fontdict=font)
        plt.ylabel(YY, fontdict=font)
        plt.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
        
        fig.savefig('./results/Density_HsTp'+'.png',dpi=300)        

class sl_data(object):
    """
    sl_data
    
    Preprocessing the sea level data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path):
        """
        Define the file path, data source, and dataset
        """
        self.filePath = path
        self.dataSource = None
        self.dataGOS = None
        self.dataGOT = None
        self.dataTide = None
        self.dataSL = None
        self.dataSurge = None
        self.dataTime = None
        self.dt = None
        self.sl = None
        
        self.readSL()
        if self.dt is not None:
                self.saveData(path)
        
    def readSL(self):
        """
        Read sea level data
        """
        try:
                self.dataGOS = scipy.io.loadmat(self.filePath+'/GOS.mat')
                self.dataTime = self.dataGOS["data"]['time'][0][0].flatten()
                dt = pd.to_datetime(self.dataTime - 719529, unit='d').round('s').to_pydatetime()
                self.dt = pd.DatetimeIndex(dt)
                self.dataSurge = self.dataGOS["data"]['zeta'][0][0].flatten()
                self.dataSource = 'IH-DATA'
        except:
                pass
        
        try:
                self.dataGOT = scipy.io.loadmat(self.filePath+'/GOT.mat')
                self.dataTime = self.dataGOT["data"]['time'][0][0].flatten()
                dt = pd.to_datetime(self.dataTime - 719529, unit='d').round('s').to_pydatetime()
                self.dt = pd.DatetimeIndex(dt)
                self.dataTide = self.dataGOT["data"]['tide'][0][0].flatten()
                self.dataSource = 'IH-DATA'
        except:
                pass
        
        try:
                Tide = pd.read_csv(self.filePath+'/Tide.csv')
                self.dataTime = Tide['Datetime'].values
                self.dt = pd.to_datetime(self.dataTime)
                self.dataTide = Tide['Tide'].values
                self.dataSource = 'CSV file'
        except:
                pass
        try:
                Surge = pd.read_csv(self.filePath+'/Surge.csv')
                self.dataTime = Surge['Datetime'].values
                self.dt = pd.to_datetime(self.dataTime)
                self.dataSurge = Surge['Surge'].values
                self.dataSource = 'CSV file'
        except:
                pass
        try:
                sl = pd.read_csv(self.filePath+'/Sealevel.csv')
                self.dataTime = sl['Datetime'].values
                self.dt = pd.to_datetime(self.dataTime)
                self.dataSL = sl['Sealevel'].values
                self.dataSource = 'CSV file'
        except:
                pass

        return 'Data loaded correctly'

    def saveData(self, path):
        """
        Save sea level data
        """
        self.sl = xr.Dataset(coords={
                'Y': self.dt.year.values,
                'M': self.dt.month.values,
                'D': self.dt.day.values,
                'h': self.dt.hour.values,
                'tide': self.dataTide,
                'surge': self.dataSurge,
                'sl': self.dataSL
                })
        self.sl.to_netcdf(path+'/sl.nc', engine='netcdf4')

class obs_data(object):
    """
    obs_data
    
    Preprocessing the observation data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self, path):
        """
        Define the file path, data source, and dataset
        """
        self.filePath = path
        self.dataSource = None
        self.data = None
        self.dataTime = None
        self.obs = None
        self.dt = None
        
        self.readObs()
        if self.dt is not None:
                self.saveData(path)
        
    def readObs(self):
        """
        Read observation data
        """
        try:
                self.data = pd.read_csv(self.filePath+'/Obs.csv')
                self.dataTime = self.data['Datetime'].values
                self.dt = pd.to_datetime(self.dataTime)
                print(self.dt)
                self.dataSource = 'CSV file'
        except:
                pass
 
        return 'Data loaded correctly'

    def saveData(self, path):
        """
        Save observation data
        """
        self.ens = xr.Dataset(coords={
                'Y': self.dt.year.values,
                'M': self.dt.month.values,
                'D': self.dt.day.values,
                'h': self.dt.hour.values,
                'Obs': self.data['Observation'].values
                })
        self.ens.to_netcdf(path+'/ens.nc', engine='netcdf4')


# class pre_config(object):
#     """
#     pre_config
    
#     Preprocessing the configuration for IH-SET.
    
#     This class reads input datasets, performs its preprocess.
#     """
#     def __init__(self, path, cfg):
#             self.path = path
#             config = xr.Dataset(coords=cfg)
#             config.to_netcdf(self.path+'/config.nc', engine='netcdf4')
#             config.close()

# class pre_wav_csv(object):
#     """
#     pre_wav_csv
    
#     Preprocessing the wave data for IH-SET.
    
#     This class reads input datasets, performs its preprocess.
#     """
#     def __init__(self, path, path_Hs, path_Tp, path_Dir):
#         """
#         Wave csv files (Copernicus, buoy, etc.)
        
#         Each csv files for Hs, Tp, Dir should contain the followings:
        
#         Datetime : YYYY-MM-DD (Shoreshop) or YYYY-MM-DDThh:mm:ss.sssZ(Copernicus)
        
#         Variable : Hs, Tp, Dir
        
#         """
#         Hs = pd.read_csv(path_Hs)
#         Tp = pd.read_csv(path_Tp)
#         Dir = pd.read_csv(path_Dir)
        
#         Time = Hs['Datetime'].values
#         dt = pd.to_datetime(Time)

#         wav = xr.Dataset(coords={
#                 'Y': dt.year.values,
#                 'M': dt.month.values,
#                 'D': dt.day.values,
#                 'h': dt.hour.values,
#                 'Hs': Hs['Hs'].values,
#                 'Tp': Tp['Tp'].values,
#                 'Dir': Dir['Dir'].values
#                 })
        
#         wav.to_netcdf(path+'/wav.nc', engine='netcdf4')
        
# class pre_sl_csv(object):
#     """
#     pre_sl_csv
    
#     Preprocessing the sealevel data for IH-SET.
    
#     This class reads input datasets, performs its preprocess.
#     """
#     def __init__(self, path, path_tide, path_surge, path_sl):
#         """
#         Wave csv files (Copernicus, buoy, etc.)
        
#         Each csv files for Hs, Tp, Dir should contain the followings:
        
#         Datetime : YYYY-MM-DD (Shoreshop) or YYYY-MM-DDThh:mm:ss.sssZ(Copernicus)
        
#         Variable : Tide, Surge, Sealevel
        
#         """
#         Tide = pd.read_csv(path_tide)
#         Surge = pd.read_csv(path_surge)
#         Sealevel = pd.read_csv(path_sl)
        
#         Time = Tide['Datetime'].values
#         dt = pd.to_datetime(Time)

#         sl = xr.Dataset(coords={
#                 'Y': dt.year.values,
#                 'M': dt.month.values,
#                 'D': dt.day.values,
#                 'h': dt.hour.values,
#                 'tide': Tide['Tide'].values,
#                 'surge': Surge['Surge'].values,
#                 'sl': Sealevel['Sealevel'].values
#                 })

#         sl.to_netcdf(path+'/sl.nc', engine='netcdf4')
        
# class pre_wav_ih(object):
#     """
#     pre_wav_ih
    
#     Preprocessing the wave data for IH-SET.
    
#     This class reads input datasets, performs its preprocess.
#     """
#     def __init__(self, path, path_GOW):
#         """
#         Wave IH-Data files
                
#         GOW file in IH-Data should contain the followings:
        
#         Datetime : ['time'] datenum for matlab
        
#         Variable : ['hs'] for Hs, ['t02'] for Tp, ['dir'] for Dir
        
#         """
#         IH_Data_GOW = scipy.io.loadmat(path_GOW)
#         Time = IH_Data_GOW["data"]['time'][0][0].flatten()
#         dates = pd.to_datetime(Time - 719529, unit='d').round('s').to_pydatetime()

#         wav = xr.Dataset(coords={
#                 'Y': ('time', [dt.year for dt in dates]),
#                 'M': ('time', [dt.month for dt in dates]),
#                 'D': ('time', [dt.day for dt in dates]),
#                 'h': ('time', [dt.hour for dt in dates]),
#                 'Hs': IH_Data_GOW["data"]['hs'][0][0].flatten(),
#                 'Tp': IH_Data_GOW["data"]['t02'][0][0].flatten(),
#                 'Dir': IH_Data_GOW["data"]['dir'][0][0].flatten()
#                 })

#         wav.to_netcdf(path+'/wav.nc', engine='netcdf4')
        
# class pre_sl_ih(object):
#     """
#     pre_sl_ih
    
#     Preprocessing the sea level data for IH-SET.
    
#     This class reads input datasets, performs its preprocess.
#     """
#     def __init__(self, path, path_GOS, path_GOT):
#         """
#         Wave IH-Data files
                
#         GOT and GOW files in IH-Data should contain the followings:
        
#         Datetime : ['time'] datenum for matlab
        
#         Variable : ['hs'] for Hs, ['t02'] for Tp, ['dir'] for Dir
        
#         """
#         IH_Data_GOS = scipy.io.loadmat(path_GOS)
#         IH_Data_GOT = scipy.io.loadmat(path_GOT)
#         Time = IH_Data_GOS["data"]['time'][0][0].flatten()
#         dates = pd.to_datetime(Time - 719529,unit='d').round('s').to_pydatetime()

#         sl = xr.Dataset(coords={
#                 'Y': ('time', [dt.year for dt in dates]),
#                 'M': ('time', [dt.month for dt in dates]),
#                 'D': ('time', [dt.day for dt in dates]),
#                 'h': ('time', [dt.hour for dt in dates]),
#                 'surge': IH_Data_GOS["data"]['zeta'][0][0].flatten(),
#                 'tide': IH_Data_GOT["data"]['tide'][0][0].flatten()
#                 })
        
#         sl.to_netcdf(path+'/sl.nc', engine='netcdf4')
        
# class pre_obs(object):    
#     """
#     pre_obs
    
#     Preprocessing the observation data for IH-SET.
    
#     This class reads input datasets, performs its preprocess.
    
#     """
#     def __init__(self, path, path_obs):
            
#         """
#         Shoreline or Orientation observation files (Shoreshop, GPS)
        
#         Each csv files for observation should contain the followings:
        
#         Datetime : YYYY-MM-DD (Shoreshop)
        
#         Variable : Observation (Shoreline or Orientation)
        
#         """
#         Obs = pd.read_csv(path_obs)

#         Time = Obs['Datetime'].values
#         dt = pd.to_datetime(Time)

#         ens = xr.Dataset(coords={
#                 'Y': dt.year.values,
#                 'M': dt.month.values,
#                 'D': dt.day.values,
#                 'h': dt.hour.values,
#                 'Obs': Obs['Observation'].values
#                 })

#         ens.to_netcdf(path+'/ens.nc', engine='netcdf4')
