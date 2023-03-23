#==============================================================================
# This script plots seasonal insolation and temperature from the transient
# HadCM3 simulation for both hemispheres. This is Fig. 3 in the paper.
#    author: Michael Erb
#    date  : 3/23/2023
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/data_assimilation')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import da_utils
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature

save_instead_of_plot = False
use_smoothed_model_output = True



#%% LOAD DATA

if use_smoothed_model_output: extra_txt = '_s'
else:                         extra_txt = ''

data_dir = '/projects/pd_lab/data/models/HadCM3B_transient21k/deglh/vn1_0/'
time_means = ['ANN','MAM','JJA','SON','DJF']

# Load coordinates and time
handle = xr.open_dataset(data_dir+'deglh.vn1_0.temp_mm_1_5m.monthly.ANN.001'+extra_txt+'.nc',decode_times=False)
lat = handle['latitude'].values
lon = handle['longitude'].values
age = -1*handle['t'].values
handle.close()

# Load insolation
insolation,temperature = {},{}
for time_mean in time_means:
    handle = xr.open_dataset(data_dir+'deglh.vn1_0.downSol_mm_TOA.monthly.'+time_mean+'.001.nc',decode_times=False)
    insolation[time_mean] = np.squeeze(handle['downSol_mm_TOA'].values)
    handle.close()
    #
    handle = xr.open_dataset(data_dir+'deglh.vn1_0.temp_mm_1_5m.monthly.'+time_mean+'.001'+extra_txt+'.nc',decode_times=False)
    temperature[time_mean] = np.squeeze(handle['temp_mm_1_5m'].values)
    handle.close()


#%% CALCULATIONS

# Find some time periods
indices_ref = np.where((age >= -50) & (age <= 50))[0]

# Remove the mean of the 100 year period from 1900-2000 and compute hemispheric means
for time_mean in time_means:
    insolation[time_mean+'_anom'] = insolation[time_mean] - np.mean(insolation[time_mean][indices_ref,:,:],axis=0)[None,:,:]
    insolation[time_mean+'_NH_mean'] = da_utils.spatial_mean(insolation[time_mean+'_anom'],lat,lon,  0,90,0,360,1,2)
    insolation[time_mean+'_SH_mean'] = da_utils.spatial_mean(insolation[time_mean+'_anom'],lat,lon,-90, 0,0,360,1,2)
    #
    temperature[time_mean+'_anom'] = temperature[time_mean] - np.mean(temperature[time_mean][indices_ref,:,:],axis=0)[None,:,:]
    temperature[time_mean+'_NH_mean'] = da_utils.spatial_mean(temperature[time_mean+'_anom'],lat,lon,  0,90,0,360,1,2)
    temperature[time_mean+'_SH_mean'] = da_utils.spatial_mean(temperature[time_mean+'_anom'],lat,lon,-90, 0,0,360,1,2)


#%%

# Compute the magnitude of the seasonal cycle at different time periods
insolation_seasonality  = insolation['JJA']  - insolation['DJF']
temperature_seasonality = temperature['JJA'] - temperature['DJF']

ind_0ka  = np.where((age >= 0)     & (age <= 1000))[0]
ind_6ka  = np.where((age >= 5500)  & (age <= 6500))[0]
ind_12ka = np.where((age >= 11000) & (age <= 12000))[0]


#%% FIGURES
plt.style.use('ggplot')

plt.figure(figsize=(12,7))
ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.Robinson()); ax1.set_global()

seasonality_diff = np.mean(temperature_seasonality[ind_12ka,:,:],axis=0) - np.mean(temperature_seasonality[ind_0ka,:,:],axis=0)
seasonality_diff_nh = da_utils.spatial_mean(seasonality_diff,lat,lon,  0,90,0,360,0,1)
seasonality_diff_sh = da_utils.spatial_mean(seasonality_diff,lat,lon,-90, 0,0,360,0,1)
seasonality_diff_cyclic,lon_cyclic = cutil.add_cyclic_point(seasonality_diff,coord=lon)
map1 = ax1.contourf(lon_cyclic,lat,seasonality_diff_cyclic,np.linspace(-10,10,21),extend='both',cmap='bwr',transform=ccrs.PlateCarree())
colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
colorbar1.ax.tick_params(labelsize=14)
ax1.set_title('Change in seasonality (JJA-DJF) for 12-0 ka ($^\circ$C)\nNH$_{mean}$='+str('{:.2f}'.format(seasonality_diff_nh))+', SH$_{mean}$='+str('{:.2f}'.format(seasonality_diff_sh)),loc='center',fontsize=20)

ax1.coastlines()
ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
ax1.spines['geo'].set_edgecolor('black')

if save_instead_of_plot:
    plt.savefig('figures/seasonality_change'+extra_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Plot insolation for different seasons
f, ax = plt.subplots(2,1,figsize=(12,12))
ax = ax.ravel()

title_txt = {'NH':'(a) Northern Hemisphere','SH':'(b) Southern Hemisphere'}

ax_twin = {}
for i,region in enumerate(['NH','SH']):
    ax[i].plot(age,insolation['JJA_'+region+'_mean'],color='tab:red', linestyle='dashed',linewidth=2,label='JJA')
    ax[i].plot(age,insolation['DJF_'+region+'_mean'],color='tab:blue',linestyle='dashed',linewidth=2,label='DJF')
    ax[i].plot(age,insolation['ANN_'+region+'_mean'],color='k',       linestyle='dashed',linewidth=2,label='Annual')
    ax[i].set_xlim(12000,0)
    ax[i].set_ylim(-30,30)
    ax[i].set_xlabel('Age (ka)',fontsize=16)
    ax[i].set_ylabel('$\Delta$Insolation ($W m^{-2}$)',fontsize=16)
    ax[i].set_title(title_txt[region]+' means',fontsize=18,loc='left')
    #if i == 0: ax[i].legend(loc='lower right',ncol=3,fontsize=16)
    #
    ax_twin[i] = ax[i].twinx()
    ax_twin[i].plot(age,temperature['JJA_'+region+'_mean'],color='tab:red', linewidth=1,label='JJA')
    ax_twin[i].plot(age,temperature['DJF_'+region+'_mean'],color='tab:blue',linewidth=1,label='DJF')
    ax_twin[i].plot(age,temperature['ANN_'+region+'_mean'],color='k',       linewidth=1,label='Annual')
    ax_twin[i].set_ylim(-5,5)
    ax_twin[i].set_ylabel('$\Delta$T ($^\circ$C)',fontsize=16)
    ax_twin[i].grid(False)
    if i == 0: ax_twin[i].legend(loc='lower right',ncol=3,fontsize=16)

f.suptitle('Insolation ($W m^{-2}$) and temperature ($^\circ$C) from the\nHadCM3 transient 21k simulation, averaged over different seasons',fontsize=18)
if save_instead_of_plot:
    plt.savefig('figures/PaperFig3_insolation_temperature_HadCM3'+extra_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Plot insolation for different seasons
f, ax = plt.subplots(2,1,figsize=(12,12))
ax = ax.ravel()

title_txt = {'NH':'(a) Northern Hemisphere','SH':'(b) Southern Hemisphere'}

for i,region in enumerate(['NH','SH']):
    ax[i].plot(age,insolation['MAM_'+region+'_mean'],color='tab:blue', linewidth=2,label='MAM')
    ax[i].plot(age,insolation['JJA_'+region+'_mean'],color='tab:green',linewidth=2,label='JJA')
    ax[i].plot(age,insolation['SON_'+region+'_mean'],color='tab:gray', linewidth=2,label='SON')
    ax[i].plot(age,insolation['DJF_'+region+'_mean'],color='tab:red',  linewidth=2,label='DJF')
    ax[i].plot(age,insolation['ANN_'+region+'_mean'],color='k',        linewidth=5,label='Annual')
    ax[i].set_xlim(12000,0)
    ax[i].set_ylim(-30,30)
    ax[i].set_xlabel('Age (ka)',fontsize=16)
    ax[i].set_ylabel('$\Delta$Insolation ($W m^{-2}$)',fontsize=16)
    ax[i].set_title(title_txt[region]+' means',fontsize=18,loc='left')
    if i == 0: ax[i].legend(loc='lower right',ncol=3,fontsize=16)

f.suptitle('Insolation ($W m^{-2}$) from the HadCM3 transient 21k simulation,\naveraged over different seasons',fontsize=18)
if save_instead_of_plot:
    plt.savefig('figures/extra_Fig3_hemispheric_insolation_HadCM3'+extra_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Plot temperature for different seasons
f, ax = plt.subplots(2,1,figsize=(12,12))
ax = ax.ravel()

title_txt = {'NH':'(a) Northern Hemisphere','SH':'(b) Southern Hemisphere'}

for i,region in enumerate(['NH','SH']):
    ax[i].plot(age,temperature['MAM_'+region+'_mean'],color='tab:blue', linewidth=1,label='MAM')
    ax[i].plot(age,temperature['JJA_'+region+'_mean'],color='tab:green',linewidth=1,label='JJA')
    ax[i].plot(age,temperature['SON_'+region+'_mean'],color='tab:gray', linewidth=1,label='SON')
    ax[i].plot(age,temperature['DJF_'+region+'_mean'],color='tab:red',  linewidth=1,label='DJF')
    ax[i].plot(age,temperature['ANN_'+region+'_mean'],color='k',        linewidth=2,label='Annual')
    ax[i].set_xlim(12000,0)
    ax[i].set_ylim(-5,1)
    ax[i].set_xlabel('Age (ka)',fontsize=16)
    ax[i].set_ylabel('$\Delta$T ($^\circ$C)',fontsize=16)
    ax[i].set_title(title_txt[region]+' means',fontsize=18,loc='left')
    if i == 0: ax[i].legend(loc='lower right',ncol=3,fontsize=16)

f.suptitle('Temperature ($^\circ$C) from the HadCM3 transient 21k simulation,\naveraged over different seasons',fontsize=18)
if save_instead_of_plot:
    plt.savefig('figures/extra_Fig3_hemispheric_temperature_HadCM3'+extra_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


