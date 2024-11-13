import scipy.io
import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# from windrose import WindroseAxes
# import seaborn as sns
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from CoastSat_handler import shoreline

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
            # self.time = np.vectorize(lambda x: np.datetime64(x))(self.time)
            self.hs = self.data['hs'].flatten()
            self.tp = self.data['tps'].flatten()
            self.dir = self.data['dir'].flatten()
            self.lat = self.data['lat'].flatten()
            self.lon = self.data['lon'].flatten()
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

        # the variables have dimension (time), we need dimensions (time, 1)

        self.hs = self.hs.reshape(-1,1)
        self.tp = self.tp.reshape(-1,1)
        self.dir = self.dir.reshape(-1,1)

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
        
        self.surge = self.surge.reshape(-1,1)
        
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

        self.tide = self.tide.reshape(-1,1)
        
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
        # self.ntrs = np.array([1])
        self.ntrs = None
        
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
            self.obs = self.obs.reshape(-1,1)
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
    
    def CoastSatR(self, epsg, sea_point, ref_points, dx, length=500):
         '''
         This function extract timeseries from CoastSat geojson MULTIPOINT output.
         '''

         if self.dataSource == 'CoastSat':
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