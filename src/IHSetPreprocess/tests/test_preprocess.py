import os
from IHSetPreprocess import IHSetPreprocess
import xarray as xr

wrkDir = os.getcwd()
path = wrkDir+'/data'

obs = IHSetPreprocess.obs_data(path)
sl = IHSetPreprocess.sl_data(path)
wav = IHSetPreprocess.wave_data(path)
