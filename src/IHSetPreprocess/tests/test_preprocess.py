import os
from IHSetPreprocess import IHSetPreprocess
import xarray as xr

wrkDir = os.getcwd()

obs = IHSetPreprocess.obs_data(wrkDir+'/data')

obs = xr.open_dataset(wrkDir+'/data/ens.nc')
print(obs['Obs'].values)

sl = IHSetPreprocess.sl_data(wrkDir+'/data')

slv = xr.open_dataset(wrkDir+'/data/sl.nc')
# print(slv['surge'].values)
# print(slv['tide'].values)
# print(slv['sl'].values)

wav = IHSetPreprocess.wave_data(wrkDir+'/data')

wav = xr.open_dataset(wrkDir+'/data/wav.nc')
print(wav['Hs'].values)
print(wav['Tp'].values)
print(wav['Dir'].values)
