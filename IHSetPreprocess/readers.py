import scipy.io
import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# from windrose import WindroseAxes
# import seaborn as sns
import numpy as np


class wave_data(object):
    """
    wave_data
    
    Preprocessing the wave data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self):
        """
        Define the file path, data source, and dataset
        """
        self.filePath = None
        self.dataSource = None
        self.data = None
        self.hs = None
        self.tp = None
        self.dir = None
        self.time = None
        self.dataSource = None
        self.lat = None
        self.lon = None
                        
    def readWaves(self, path):
        """
        Read wave data
        """
        
        self.filePath = path

        try:
            self.data = scipy.io.loadmat(self.filePath)
            self.dataTime = self.data['time'].flatten()
            self.time = pd.to_datetime(self.dataTime - 719529, unit='d').round('s').to_pydatetime()
            self.time = np.vectorize(lambda x: np.datetime64(x))(self.time)
            self.hs = self.data['hs'].flatten()
            self.tp = self.data['tps'].flatten()
            self.dir = self.data['dir'].flatten()
            self.lat = self.data['lat'].flatten()[0]
            self.lon = self.data['lon'].flatten()[0]
            self.dataSource = 'IH-DATA'
        except:
            pass

        try:
            self.data = pd.read_csv(self.filePath)
            self.time = pd.to_datetime(self.data['Datetime'])
            self.hs = self.data['Hs']
            self.tp = self.data['Tp']
            self.dir = self.data['Dir']
            self.dataSource = 'CSV file'
        except:
            pass

        if self.dataSource == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'
            
    # def HsRose(self):
    #     """
    #     Plotting HsRose
    #     """
    #     plt.rcParams.update({'font.family': 'serif'})
    #     plt.rcParams.update({'font.size': 7})
    #     plt.rcParams.update({'font.weight': 'bold'})

    #     fig = plt.figure(figsize=(6.5, 5), dpi=300, linewidth=5, edgecolor="#04253a")
    #     ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
    #     fig.add_axes(ax)
        
    #     data = pd.DataFrame({'X': self.dir, 'Y': self.hs})
    #     data = data.dropna()
    #     ax.bar(data['X'], data['Y'], normed=True, bins=[0, 0.5, 1, 1.5, 2, 2.5], opening=0.8, edgecolor='white')
    #     ax.set_legend(title=r"$Hs \, (m)$", loc='best')
        
    #     fig.savefig('./results/Rose_Wave'+'.png',dpi=300)
        
    # def TpRose(self):
    #     """
    #     Plotting TpRose
    #     """
    #     plt.rcParams.update({'font.family': 'serif'})
    #     plt.rcParams.update({'font.size': 7})
    #     plt.rcParams.update({'font.weight': 'bold'})

    #     fig = plt.figure(figsize=(6.5, 5), dpi=300, linewidth=5, edgecolor="#04253a")
    #     ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
    #     fig.add_axes(ax)
        
    #     data = pd.DataFrame({'X': self.wav['Dir'], 'Y': self.wav['Tp']})
    #     data = data.dropna()
    #     ax.bar(data['X'], data['Y'], normed=True, bins=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0], opening=0.8, edgecolor='white')
    #     ax.set_legend(title=r"$Tp \, (sec)$", loc='best')
        
    #     fig.savefig('./results/Rose_Wave'+'.png',dpi=300)
        
    # def densityHsTp(self):
    #     """
    #     Plotting densityHsTp
    #     """
    #     plt.rcParams.update({'font.family': 'serif'})
    #     plt.rcParams.update({'font.size': 7})
    #     plt.rcParams.update({'font.weight': 'bold'})
    #     font = {'family': 'serif',
    #             'weight': 'bold',
    #             'size': 8}

    #     XX = 'Hs'
    #     YY = 'Tp'
    #     data = pd.DataFrame({'X': self.wav[XX], 'Y': self.wav[YY]})
    #     data = data.dropna()

    #     fig = plt.figure(figsize=(5, 5), dpi=300, linewidth=5, edgecolor="#04253a")

    #     sns.kdeplot(data=data, x='X', y='Y', cmap='Blues', fill=True, thresh=0, levels=20)

    #     plt.xlim([np.floor(np.min(data['X'])),np.floor(np.max(data['X']))])
    #     plt.ylim([np.floor(np.min(data['Y'])),np.floor(np.max(data['Y']))])
    #     plt.xlabel(XX, fontdict=font)
    #     plt.ylabel(YY, fontdict=font)
    #     plt.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
        
    #     fig.savefig('./results/Density_HsTp'+'.png',dpi=300)        

class sl_data(object):
    """
    sl_data
    
    Preprocessing the sea level data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self):
        """
        Define the file path, data source, and dataset
        """
        self.dataSource_tide = None
        self.dataSource_surge = None
        self.tide = None
        self.slr = None
        self.surge = None
        self.time_tide = None
        self.time_surge = None
        self.lat_tide = None
        self.lon_tide = None
        self.lat_surge = None
        self.lon_surge = None
        
    def readSurge(self, filePath):
        """
        Read sea level data
        """
        try:
            GOS = scipy.io.loadmat(filePath)
            Time = GOS['time'].flatten()
            self.time_surge = pd.to_datetime(Time - 719529, unit='d').round('s').to_pydatetime()
            self.time_surge = np.vectorize(lambda x: np.datetime64(x))(self.time_surge)
            self.surge = GOS['zeta'].flatten()
            self.lat_surge = GOS['lat_zeta'].flatten()[0]
            self.lon_surge = GOS['lon_zeta'].flatten()[0]
            self.dataSource_surge = 'IH-DATA'
        except:
                pass
        
        try:
            Surge = pd.read_csv(self.filePath)
            Time = Surge['Datetime'].values
            self.time_surge = pd.to_datetime(Time)
            self.dataTide = Surge['Tide'].values
            self.dataSource_surge = 'CSV file'
        except:
                pass
        
        if self.dataSource_surge == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'

    def readTide(self, filePath):
        """
        Read sea level data
        """
        try:
            GOT = scipy.io.loadmat(filePath)
            Time = GOT['time'].flatten()
            self.time_tide = pd.to_datetime(Time - 719529, unit='d').round('s').to_pydatetime()
            self.time_tide = np.vectorize(lambda x: np.datetime64(x))(self.time_tide)
            self.tide = GOT['tide'].flatten()
            self.lat_tide = GOT['lat_tide'].flatten()[0]
            self.lon_tide = GOT['lon_tide'].flatten()[0]
            self.dataSource_tide = 'IH-DATA'
        except:
                pass
        
        try:
            Tide = pd.read_csv(self.filePath)
            Time = Tide['Datetime'].values
            self.time_tide = pd.to_datetime(Time)
            self.dataTide = Tide['Tide'].values
            self.dataSource_tide = 'CSV file'
        except:
            pass
        
        if self.dataSource_tide == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'
        
    # def readSLR(self, filePath):
    #     """
    #     Read sea level data
    #     """
        
    #     if self.dataSource_slr == None:
    #         return 'Wrong data format'
    #     else:
    #         return 'Data loaded correctly'

class obs_data(object):
    """
    obs_data
    
    Preprocessing the observation data for IH-SET.
    
    This class reads input datasets, performs its preprocess.
    """
    def __init__(self):
        """
        Define the file path, data source, and dataset
        """
        self.dataSource = None
        self.time_obs = None
        self.obs = None
        
    def readObs(self, path):
        """
        Read observation data
        """
        self.filePath = path
        try:
                data = pd.read_csv(self.filePath)
                Time = data['Datetime'].values
                self.time_obs = pd.to_datetime(Time)
                self.obs = data['Obs']
                self.dataSource = 'CSV file'
        except:
                pass
        
        if self.dataSource == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'

