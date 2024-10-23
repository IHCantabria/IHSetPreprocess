import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import seaborn as sns
import numpy as np
from datetime import datetime
from IHSetPreprocess import IHSetPreprocess

## Test Preprocess the configuration
config = {'dt': 3,                  # [hours]
          'depth': 20,              # Water depth [m] (MD, TU, JR, LIM)
          'D50': .3e-3,             # Median grain size [m] (MD, TU, SF, JR, LIM)
          'bathy_angle': 90-7.2,    # Bathymetry mean orientation [deg N] (MD, TU, JR, LIM)
          'break_type': 'spectral', # Breaking type (spectral or linear) (MD, TU)
          'xc' : 255,               # Cross-shore distance from shore to closure depth [m] (JR)
          'hc' : 6,                 # Depth of closure [m] (JR)                    
          'Hberm': 1,               # Berm height [m] (MD, JR)
          'flagP': 3,               # Parameter Proportionality (MD)
          'mf' : 0.02,              # Proflie slope (LIM)
          'vlt': 0,                 # Longterm trend [m] (JA-cross)
          'BeachL': 3200           # Beach Length [m] (TU)]
          }

wrkDir = os.getcwd()
IHSetPreprocess.pre_config(wrkDir+'/data', config)

## Test Preprocess the Wave csv files (Copernicus, ShoreShop, etc.)
IHSetPreprocess.pre_wav_csv(wrkDir+'/data', wrkDir+'/data/Hs.csv', wrkDir+'/data/Tp.csv', wrkDir+'/data/Dir.csv')

## Test Preprocess the sea level csv files (Surge, Sea Level Rise, Tide, etc.)
IHSetPreprocess.pre_sl_csv(wrkDir+'/data', wrkDir+'/data/Tide.csv', wrkDir+'/data/Surge.csv', wrkDir+'/data/Sealevel.csv')

## Test Preprocess the IH-Data(Wave)
IHSetPreprocess.pre_wav_ih(wrkDir+'/data', wrkDir+'/data/GOW_Example.mat')

## Test Preprocess the IH-Data(Tide, Surge)
IHSetPreprocess.pre_sl_ih(wrkDir+'/data', wrkDir+'/data/GOS_Example.mat', wrkDir+'/data/GOT_Example.mat')

## Test Preprocess the observation data (Shoreline or Orientation)
IHSetPreprocess.pre_obs(wrkDir+'/data', wrkDir+'/data/Obs.csv')

##################################################################################################################
############################################# Statistical Calculation ############################################
wav = xr.open_dataset(wrkDir+'/data/wav.nc') # Apply all variables in nc files (wav.nc, obs.nc, sl.nc)
Hs = wav['Hs'].values
Hs = Hs[~np.isnan(Hs)]
mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
wav['time'] = pd.to_datetime(mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values))

Hs_mean = Hs.mean()                          # Mean
Hs_rms = np.sqrt(np.mean(Hs**2))             # RMS
Hs_std = Hs.std()                            # Standard deviation
Hs_max = Hs.max()                            # Maxium
Hs_min = Hs.min()                            # Minimum

print(f"Mean Hs: {Hs_mean}")
print(f"RMS Hs: {Hs_rms}")
print(f"Standard deviation Hs: {Hs_std}")
print(f"Maxium Hs: {Hs_max}")
print(f"Minimum Hs: {Hs_min}")

##################################################################################################################
################################################ Wave Time Series ################################################
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

wav = xr.open_dataset(wrkDir+'/data/wav.nc')
mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)

fig, ax = plt.subplots(3 , 1, figsize=(10, 3), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [1.5, 1.5, 1.5]})
ax[0].scatter(time, wav['Hs'], s = 1, c = 'grey', label = 'Hs [m]')
ax[0].set_xlim([time[0], time[-1]])
ax[0].set_ylim([0,np.ceil(np.max(wav['Hs']))])
ax[0].set_ylabel('Hs [m]', fontdict=font)
ax[0].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

ax[1].scatter(time, wav['Tp'], s = 1, c = 'grey', label = 'Hs [m]')
ax[1].set_xlim([time[0], time[-1]])
ax[1].set_ylim([0,np.ceil(np.max(wav['Tp']))])
ax[1].set_ylabel('Tp [sec]', fontdict=font)
ax[1].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

ax[2].scatter(time, wav['Dir'], s = 1, c = 'grey', label = 'Hs [m]')
ax[2].set_xlim([time[0], time[-1]])
ax[2].set_ylim([0,360])
ax[2].set_ylabel('Dir [deg]', fontdict=font)
ax[2].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/TimeSeries_Wave'+'.png',dpi=300)
# plt.show()

##################################################################################################################
################################################ Wave Joint Dist. ################################################
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

XX = 'Tp'
YY = 'Dir'
data = pd.DataFrame({'X': wav[XX], 'Y': wav[YY]})
data = data.dropna()

fig = plt.figure(figsize=(5, 5), dpi=300, linewidth=5, edgecolor="#04253a")

### Choose plot option
# sns.kdeplot(data=data, x='X', y='Y', cmap='Blues', fill=True, thresh=0, levels=20)
sns.scatterplot(data=data, x='X', y='Y', alpha=0.5)
# plt.hexbin(data['X'], data['Y'], gridsize=15, cmap='Blues')

plt.xlim([np.floor(np.min(data['X'])),np.floor(np.max(data['X']))])
plt.ylim([np.floor(np.min(data['Y'])),np.floor(np.max(data['Y']))])
plt.xlabel(XX, fontdict=font)
plt.ylabel(YY, fontdict=font)
plt.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
fig.savefig('./results/Distribution_Wave'+'.png',dpi=300)

##################################################################################################################
################################################## Rose Diagram ##################################################
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

fig = plt.figure(figsize=(6.5, 5), dpi=300, linewidth=5, edgecolor="#04253a")
ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
fig.add_axes(ax)


## Option 1
# data = pd.DataFrame({'X': wav['Dir'], 'Y': wav['Hs']})
# data = data.dropna()
# ax.bar(data['X'], data['Y'], normed=True, bins=[0, 0.5, 1, 1.5, 2, 2.5], opening=0.8, edgecolor='white')
# ax.set_legend(title=r"$Hs \, (m)$", loc='best')

## Option 2
# data = pd.DataFrame({'X': wav['Dir'], 'Y': wav['Tp']})
# data = data.dropna()
# ax.bar(data['X'], data['Y'], normed=True, bins=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0], opening=0.8, edgecolor='white')
# ax.set_legend(title=r"$Tp \, (sec)$", loc='best')

## Option 3
data = pd.DataFrame({'X': wav['Dir'], 'Y': wav['Hs'].values**2 * wav['Tp'].values})
data = data.dropna()
ax.bar(data['X'], data['Y'], normed=True, bins=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0], opening=0.8, edgecolor='white')
ax.set_legend(title=r"$Hs^2 \, Tp \, (m^2s)$", loc='best')

fig.savefig('./results/Rose_Wave'+'.png',dpi=300)

##################################################################################################################
################################ Observation (Shoreline, Orientation) Time Series ################################
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

ens = xr.open_dataset(wrkDir+'/data/ens.nc')
mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
time = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

fig, ax = plt.subplots(1 , 1, figsize=(10, 3), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [1.5]})
ax.scatter(time, ens['Obs'], s = 1, c = 'grey', label = 'Observation')
ax.set_xlim([time[0], time[-1]])
ax.set_ylim([np.floor(np.min(ens['Obs'])),np.ceil(np.max(ens['Obs']))])
ax.set_ylabel('Obs.', fontdict=font)
ax.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/TimeSeries_Obs'+'.png',dpi=300)

##################################################################################################################
############################################### Surge Time Series ################################################
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

slv = xr.open_dataset(wrkDir+'/data/sl.nc')
mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
time = mkTime(slv['Y'].values, slv['M'].values, slv['D'].values, slv['h'].values)

fig, ax = plt.subplots(1 , 1, figsize=(10, 3), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [1.5]})
ax.scatter(time, slv['surge'], s = 1, c = 'grey', label = 'Surge')
ax.set_xlim([time[0], time[-1]])
ax.set_ylim([np.floor(np.min(slv['surge'])),np.ceil(np.max(slv['surge']))])
ax.set_ylabel('Surge [m]', fontdict=font)
ax.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/TimeSeries_Surge'+'.png',dpi=300)

##################################################################################################################
################################################ Tide Time Series ################################################
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

slv = xr.open_dataset(wrkDir+'/data/sl.nc')
mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
time = mkTime(slv['Y'].values, slv['M'].values, slv['D'].values, slv['h'].values)

fig, ax = plt.subplots(1 , 1, figsize=(10, 3), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [1.5]})
ax.scatter(time, slv['tide'], s = 1, c = 'grey', label = 'Tide')
ax.set_xlim([time[0], time[-1]])
ax.set_ylim([np.floor(np.min(slv['tide'])),np.ceil(np.max(slv['tide']))])
ax.set_ylabel('Tide [m]', fontdict=font)
ax.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/TimeSeries_Tide'+'.png',dpi=300)