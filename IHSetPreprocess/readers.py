import scipy.io
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from .CoastSat_handler import shoreline
import xarray as xr
from matplotlib import pyplot as plt
from windrose import WindroseAxes
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from pyproj import CRS, Transformer
from IHSetUtils.geometry import abs_pos


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
        self.epsg = None
                        
    def readWaves(self, path):
        """
        Read wave data
        """
        
        self.filePath = path

        try:
            self.data = scipy.io.loadmat(self.filePath)
            self.dataTime = self.data['time'].flatten()
            self.time = pd.to_datetime(self.dataTime - 719529, unit='d').round('s').to_pydatetime()
            # self.time = np.vectorize(lambda x: np.datetime64(x))(self.time)
            self.hs = self.data['hs'].flatten()
            self.tp = self.data['tps'].flatten()
            self.dir = self.data['dir'].flatten()
            self.lat = self.data['lat'].flatten()
            self.lon = self.data['lon'].flatten()
            self.dataSource = 'IH-DATA'
            self.epsg = 4326
        except:
            pass

        try:
            self.data = pd.read_csv(self.filePath)
            self.time = pd.to_datetime(self.data['time'].values)
            self.hs = self.data['Hs'].values
            self.tp = self.data['Tp'].values
            self.dir = self.data['Dir'].values
            self.dataSource = 'CSV file'
        except:
            pass

        try:
            self.data = xr.open_dataset(self.filePath)
            self.time = pd.to_datetime(self.data.time.values)
            self.hs = self.data.VHM0.values.flatten()
            self.tp = self.data.VTPK.values.flatten()
            self.dir = self.data.VMDR.values.flatten()
            self.lat = self.data.latitude.values
            self.lon = self.data.longitude.values
            self.epsg = 4326
            self.dataSource = 'Copernicus'
        except:
            pass

        # the variables have dimension (time), we need dimensions (time, 1)
        self.hs = self.hs.reshape(-1,1)
        self.tp = self.tp.reshape(-1,1)
        self.dir = self.dir.reshape(-1,1)

        if self.dataSource == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'
    
    def add_coords(self, path):
        """ Add coordinates to the handler """

        coords = pd.read_csv(path)
        self.lat = coords.lat.values
        self.lon = coords.lon.values
        self.epsg = coords.epsg.values[0]

    def HsRose(self):
        """
        Plotting HsRose
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig = plt.figure(figsize=(5, 3), dpi=200, linewidth=5, edgecolor="#04253a")
        ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
        fig.add_axes(ax)
        dd = np.reshape(self.dir, (len(self.dir)))
        hhs = np.reshape(self.hs, (len(self.hs)))
        data = pd.DataFrame({'X': dd, 'Y': hhs})
        data = data.dropna()
        cmap = plt.cm.jet
        ax.bar(data['X'], data['Y'], normed=True, bins=[0, 0.5, 1, 1.5, 2, 2.5], opening=0.8, edgecolor='white', cmap=cmap)
        # ax.set_legend(title=r"$Hs \, (m)$", loc='best')
        legend = ax.set_legend(title=r"$H_s [m]$", loc='center left', bbox_to_anchor=(1.2, 0.5))
        plt.setp(legend.get_texts(), fontsize='x-small')
        plt.tight_layout()
        plt.show()
        
    def TpRose(self):
        """
        Plotting TpRose
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig = plt.figure(figsize=(5.5, 3), dpi=200, linewidth=5, edgecolor="#04253a")
        ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
        fig.add_axes(ax)

        dd = np.reshape(self.dir, (len(self.dir)))
        ttp = np.reshape(self.tp, (len(self.tp)))
        
        data = pd.DataFrame({'X': dd, 'Y': ttp})
        data = data.dropna()
        cmap = plt.cm.jet
        ax.bar(data['X'], data['Y'], normed=True, bins=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0], opening=0.8, edgecolor='white', cmap=cmap)
        legend = ax.set_legend(title=r"$T_p [s]$", loc='center left', bbox_to_anchor=(1.2, 0.5))
        plt.setp(legend.get_texts(), fontsize='x-small')
        plt.tight_layout()
        plt.show()

    
    def densityHsTpFastScatter(self):
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})
        font = {'family': 'serif', 'weight': 'bold', 'size': 8}

        # Prepare data
        XX = np.reshape(self.hs, (len(self.hs)))
        YY = np.reshape(self.tp, (len(self.tp)))
        data = pd.DataFrame({'X': XX, 'Y': YY})
        data = data.dropna()

        # Create a 2D histogram for density estimation
        xedges = np.linspace(data['X'].min(), data['X'].max(), 100)  # Adjust number of bins here
        yedges = np.linspace(data['Y'].min(), data['Y'].max(), 100)  # Adjust number of bins here
        histogram, xedges, yedges = np.histogram2d(data['X'], data['Y'], bins=(xedges, yedges))
        
        # Get bin indices for each point
        xidx = np.digitize(data['X'], xedges) - 1  # -1 to convert to 0-based index
        yidx = np.digitize(data['Y'], yedges) - 1
        # Handle edge cases where points fall on the right edge
        xidx[xidx == histogram.shape[0]] -= 1
        yidx[yidx == histogram.shape[1]] -= 1

        # Map each point to its density value
        point_density = histogram[xidx, yidx]

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        scatter = ax.scatter(data['X'], data['Y'], c=point_density, cmap='turbo', s=1, edgecolor=None, linewidths=0, alpha=1)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Density')

        # Set limits and labels
        plt.xlim([np.floor(data['X'].min()), np.ceil(data['X'].max())])
        plt.ylim([np.floor(data['Y'].min()), np.ceil(data['Y'].max())])

        plt.xlabel(r'$H_s [m]$', fontdict=font)
        plt.ylabel(r'$T_p [s]$', fontdict=font)
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def wave_timeseries(self):
        """
        Plotting wave timeseries
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig, ax = plt.subplots(3, 1, figsize=(6, 3), dpi=200, linewidth=5, edgecolor="#04253a")

        ax[0].plot(self.time, self.hs, color='blue', linewidth=0.5)
        ax[0].set_ylabel(r'$H_s [m]$')
        ax[0].grid(True)
        ax[0].set_facecolor((0, 0, 0, 0.15))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax[1].plot(self.time, self.tp, color='red', linewidth=0.5)
        ax[1].set_ylabel(r'$T_p [s]$')
        ax[1].grid(True)
        ax[1].set_facecolor((0, 0, 0, 0.15))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax[2].scatter(self.time, self.dir, s = 0.3, color='black')
        ax[2].set_ylabel(r'$Dir [°]$')
        ax[2].grid(True)
        ax[2].set_facecolor((0, 0, 0, 0.15))
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        dir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
        ax[2].set_yticks(np.arange(0, 361, 45))
        ax[2].set_yticklabels(dir_labels)
        plt.tight_layout()
        plt.show()

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
        self.time_slr = None
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
            # self.time_surge = np.vectorize(lambda x: np.datetime64(x))(self.time_surge)
            self.surge = GOS['zeta'].flatten()
            self.lat_surge = GOS['lat_zeta'].flatten()
            self.lon_surge = GOS['lon_zeta'].flatten()
            self.dataSource_surge = 'IH-DATA'
            self.surge = self.surge.reshape(-1,1)
        except:
                pass
        
        try:
            # timer = np.vectorize(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            Surge = pd.read_csv(filePath)
            self.time_surge = Surge['time'].values
            self.time_surge = pd.to_datetime(self.time_surge)
            self.surge = Surge['surge'].values
            self.dataSource_surge = 'CSV file'
            self.surge = self.surge.reshape(-1,1)
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
            # self.time_tide = np.vectorize(lambda x: np.datetime64(x))(self.time_tide)
            self.tide = GOT['tide'].flatten()
            self.lat_tide = GOT['lat_tide'].flatten()
            self.lon_tide = GOT['lon_tide'].flatten()
            self.dataSource_tide = 'IH-DATA'
            self.tide = self.tide.reshape(-1,1)
        except:
                pass
        
        try:
        # datetime format 'yyyy-mm-dd hh:mm:ss'
            # timer = np.vectorize(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            Tide = pd.read_csv(filePath)
            self.time_tide = Tide['time'].values
            self.time_tide = pd.to_datetime(self.time_tide)
            self.tide = Tide['tide'].values
            self.dataSource_tide = 'CSV file'
            self.tide = self.tide.reshape(-1,1)
        except:
            pass
        
        if self.dataSource_tide == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'
        
    def sl_timeseries(self):
        """
        Plotting sea level timeseries
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig, ax = plt.subplots(figsize=(8, 3), dpi=200, linewidth=5, edgecolor="#04253a")

        try:
            ax.plot(self.time_tide, self.tide, color='blue', linewidth=0.5, label='Tide')
        except:
            pass
        try:
            ax.plot(self.time_surge, self.surge, color='red', linewidth=0.5, label='Surge')
        except:
            pass

        ax.set_ylabel('Sea level [m]')
        ax.grid(True)
        ax.set_facecolor((0, 0, 0, 0.15))
        ax.legend()

        #Set the x-axis ticks labels as 'YYYY'
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.show()

    def sl_histogram(self):
        """
        Plotting sea level histogram
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})
        font = {'family': 'serif',
                'weight': 'bold',
                'size': 8}

        fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=200, linewidth=5, edgecolor="#04253a")

        try:
            sns.histplot(self.tide, bins=20, kde=True, color='blue', alpha=0.5, label='Tide', stat='density', ax=ax[0])
            ax[0].set_xlabel('Sea level [m]', fontdict=font)
            ax[0].set_ylabel('Density', fontdict=font)
            ax[0].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
            ax[0].legend()
        except:
            pass
        try:
            sns.histplot(self.surge, bins=20, kde=True, color='red', alpha=0.5, label='Surge', stat='density', ax=ax[1])
            ax[1].set_xlabel('Sea level [m]', fontdict=font)
            ax[1].set_ylabel('Density', fontdict=font)
            ax[1].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
            ax[1].legend()
        except:
            pass

        plt.tight_layout()
        plt.show()

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
        # self.ntrs = np.array([1])
        self.ntrs = None
        self.intersections = None
        self.xi = None
        self.yi = None
        self.xf = None
        self.yf = None
        self.phi = None
        self.epsg = None
        
    def readObs(self, path):
        """
        Read observation data
        """
        self.filePath = path

        try:
            data = pd.read_csv(self.filePath)
            Time = data['time'].values
            x = data['x'].values
            y = data['y'].values
            time_shores = pd.to_datetime(Time)
            time_obs = np.unique(time_shores)
            self.shores = {}
            for i in range(len(time_obs)):
                mask = time_shores == time_obs[i]
                xx, yy = x[mask], y[mask]
                xxf, yyf = filter_nans(xx, yy)
                self.shores[str(i)] = {}
                self.shores[str(i)]['x'] = xxf
                self.shores[str(i)]['y'] = yyf
            
            self.time_obs = time_obs
            self.dataSource = 'CSV file (time, x, y)'

        except:
                pass
        
        if self.dataSource == None:
            try:
                data = pd.read_csv(self.filePath)
                Time = data['time'].values
                self.time_obs = pd.to_datetime(Time)
                #Now we remove the time column
                data = data.drop(columns=['time'])
                self.obs = np.zeros((len(data), len(data.columns)))
                for i, key in enumerate(data.keys()):
                    self.obs[:, i] = data[key].values
                self.dataSource = 'CSV file (transects)'
            except:
                    pass
        
        try:
            geo_data = gpd.read_file(self.filePath)
            coords = geo_data['geometry'].apply(lambda geom: [point.coords[:][0] for point in geom.geoms])
            self.time_obs = pd.to_datetime(geo_data['date'])
            shores = {}
            for i in range(len(coords)):
                shores[str(i)] = {}
                shores[str(i)]['x'] = np.array([coords[i][j][0] for j in range(len(coords[i]))])
                shores[str(i)]['y'] = np.array([coords[i][j][1] for j in range(len(coords[i]))])
            self.shores = shores
            self.dataSource = 'CoastSat'
        except Exception as e:
                print(e)
                pass
        
        if self.dataSource == None:
            return 'Wrong data format'
        else:
            return 'Data loaded correctly'
        
    def add_obs_coords(self, path):
        """ Add observation coordinates to the handler """
        obs_coords = pd.read_csv(path)
        self.xi = obs_coords.xi.values
        self.yi = obs_coords.yi.values
        self.xf = obs_coords.xf.values
        self.yf = obs_coords.yf.values
        # Lets calculate the phi
        alpha = np.arctan2(self.yf - self.yi, self.xf - self.xi)
        self.phi = np.rad2deg(alpha)
        self.epsg = obs_coords.epsg.values[0]
        self.ntrs = len(self.xi)
        self.generate_shores_csv()

    def generate_shores_csv(self):
        # self.xi, self.yi, self.obs, self.epsg
        '''
        Generate the shores list from csv input with transects
        '''
        
        self.shores = {}
        for i in range(len(self.time_obs)):
            self.shores[str(i)] = {}
            x, y = abs_pos(self.xi, self.yi, np.deg2rad(self.phi), self.obs[i, :])
            self.shores[str(i)]['x'] = x
            self.shores[str(i)]['y'] = y


    
    def CoastSatR(self, epsg, sea_point, ref_points, dx, length=500):
         '''
         This function extract timeseries from CoastSat geojson MULTIPOINT output.
         '''
        domain = shoreline(self.shores, self.time_obs, epsg = epsg)
        domain.setDomain(sea_point, 'draw', dx, refPoints=ref_points)
        domain.setTransects(length)

        transects = []
        for i in range(len(domain.trs.xi)):
            transects.append({'xi': domain.trs.xi[i], 'yi': domain.trs.yi[i], 'xf': domain.trs.xf[i], 'yf': domain.trs.yf[i]})
        
        dists = find_intersections(self.shores, transects)
        intersections = find_intersections2(self.shores, transects)

        self.obs = dists
        self.ntrs = len(transects)
        self.intersections = intersections
        self.xi = domain.trs.xi
        self.yi = domain.trs.yi
        self.xf = domain.trs.xf
        self.yf = domain.trs.yf
        self.phi = domain.trs.phi
        self.epsg = domain.epsg
        # print(f"flag_f: {domain.flag_f}")
        
    def obs_timeseries(self):
        """
        Plotting observation timeseries
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})

        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, linewidth=5, edgecolor="#04253a")

        mn = self.ntrs

        colors = [mcolors.to_rgba(color) for color in plt.cm.turbo(np.linspace(0, 1, mn))]

        for i in range(mn):
            ax.plot(self.time_obs, self.obs[:,i], color=colors[i], linewidth=0.8)

        ax.set_ylabel('Shoreline position [m]')
        ax.grid(True)
        ax.set_facecolor((0, 0, 0, 0.15))

        # Adiciona a colorbar
        cmap = plt.cm.jet
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', location = 'top', shrink=0.5)

        # Define os ticks e os rótulos dos ticks da colorbar
        ticks = [0, 1]
        cbar.set_ticks(ticks)  # Define os locais dos ticks
        cbar.set_ticklabels(['Transect 1', f'Transect {mn}'])

        #Set the x-axis ticks labels as 'YYYY'
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()
    
    def obs_histogram(self):
        """
        Plotting observation histogram
        """
        plt.rcParams.update({'font.family': 'serif'})
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.weight': 'bold'})
        font = {'family': 'serif',
                'weight': 'bold',
                'size': 8}
        
        average_obs = np.nanmean(self.obs, axis=1)

        fig, ax = plt.subplots(figsize=(3.5, 3), dpi=200, linewidth=5, edgecolor="#04253a")

        sns.histplot(average_obs, bins=20, kde=True, color='blue', alpha=0.5, label='Average', stat='density', ax=ax)
        plt.xlabel('Shoreline position [m]', fontdict=font)
        plt.ylabel('Density', fontdict=font)
        plt.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
        plt.tight_layout()
        plt.show()

def find_intersections2(obs_shores, transects, gap_threshold=100):
    results = {}
    
    for time_key, shore_data in obs_shores.items():
        # Dividir a linha da costa em segmentos
        segments = split_segments(shore_data['x'], shore_data['y'], gap_threshold)
        
        transect_results = []

        for transect in transects:
            transect_line = LineString([(transect['xi'], transect['yi']), (transect['xf'], transect['yf'])])
            intersection_found = False

            # Iterar sobre cada segmento e verificar a interseção
            for segment in segments:
                intersection = segment.intersection(transect_line)

                if not intersection.is_empty:
                    intersection_found = True
                    # Se houver interseção, extrair as coordenadas do ponto de interseção
                    if isinstance(intersection, Point):
                        transect_results.append((intersection.x, intersection.y))
                    elif isinstance(intersection, LineString):
                        # Caso seja uma linha (não um ponto), extrair a coordenada média (pode ser ajustado conforme necessário)
                        coords = list(intersection.coords)
                        mean_x = np.mean([coord[0] for coord in coords])
                        mean_y = np.mean([coord[1] for coord in coords])
                        transect_results.append((mean_x, mean_y))
                    break

            # Se nenhum segmento interceptar o transecto, registrar NaN
            if not intersection_found:
                transect_results.append(np.nan)

        results[time_key] = transect_results

    return results

def split_segments(x, y, gap_threshold=100):
    segments = []
    current_segment = [(x[0], y[0])]

    for i in range(1, len(x)):
        dist = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        if dist > gap_threshold:
            segments.append(LineString(current_segment))
            current_segment = [(x[i], y[i])]
        else:
            current_segment.append((x[i], y[i]))

    if current_segment:
        segments.append(LineString(current_segment))

    return segments

# Função para calcular as interseções e gerar a matriz de saída
def find_intersections(obs_shores, transects, gap_threshold=100):
    # Número de tempos e transectos
    num_times = len(obs_shores)
    num_transects = len(transects)

    # Inicializando a matriz de resultados como NaN
    results = np.full((num_times, num_transects), np.nan)

    # Iterar sobre as keys do dicionário em ordem crescente
    for time_key in sorted(obs_shores.keys(), key=int):
        time_idx = int(time_key)  # Índice correspondente ao tempo
        shore_data = obs_shores[time_key]

        # Dividir a linha da costa em segmentos
        segments = split_segments(shore_data['x'], shore_data['y'], gap_threshold)
        
        for transect_idx, transect in enumerate(transects):
            transect_line = LineString([(transect['xi'], transect['yi']), (transect['xf'], transect['yf'])])
            intersection_found = False

            # Iterar sobre cada segmento e verificar a interseção
            for segment in segments:
                intersection = segment.intersection(transect_line)

                if not intersection.is_empty:
                    intersection_found = True
                    # Se houver interseção, calcular a posição ao longo do transecto
                    if isinstance(intersection, Point):
                        intersect_point = intersection
                    elif isinstance(intersection, LineString):
                        # Caso seja uma linha (não um ponto), considerar o ponto médio da linha de interseção
                        coords = list(intersection.coords)
                        mean_x = np.mean([coord[0] for coord in coords])
                        mean_y = np.mean([coord[1] for coord in coords])
                        intersect_point = Point(mean_x, mean_y)

                    # Calcular a posição ao longo do transecto em relação à origem
                    dist_along_transect = transect_line.project(intersect_point)
                    results[time_idx, transect_idx] = dist_along_transect
                    break

    return results

def filter_nans(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]