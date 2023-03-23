#=============================================================================
# This script compares global mean temperatures in reconstructions using
# different assumptions about the seasonality of proxy records.  It is Fig. 9
# in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/utils')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import utils_general as utils_DA
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from scipy import stats

save_instead_of_plot = False
remove_0_1ka = True


#%% LOAD DATA

# Pick the reconstruction to analyze
filenames = {}; colors = {}
filenames['author_interp'] = 'holocene_reconstruction.nc';                                            colors['author_interp'] = 'k'
filenames['annual']        = 'holocene_recon_2022-03-30_16:43:54.528687_prescribed_season_annual.nc'; colors['annual']        = 'tab:olive'
filenames['summer']        = 'holocene_recon_2022-03-30_16:43:47.132370_prescribed_season_summer.nc'; colors['summer']        = 'tab:red'
filenames['winter']        = 'holocene_recon_2022-03-30_16:44:55.579159_prescribed_season_winter.nc'; colors['winter']        = 'tab:blue'
filenames['jja']           = 'holocene_recon_2022-03-30_16:44:38.885778_prescribed_season_jja.nc';    colors['jja']           = 'tab:orange'
filenames['djf']           = 'holocene_recon_2022-03-30_16:44:59.921494_prescribed_season_djf.nc';    colors['djf']           = 'tab:brown'

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+filenames['author_interp'],decode_times=False)
lat     = handle['lat'].values
lon     = handle['lon'].values
ages_da = handle['ages'].values
handle.close()

experiments = list(filenames.keys())
recon_gmt,recon_mean,prior_gmt,proxy_metadata  = {},{},{},{}
for experiment in experiments:
    print('Loading data:',experiment)
    handle = xr.open_dataset(recon_dir+'results/'+filenames[experiment],decode_times=False)
    recon_gmt[experiment]  = handle['recon_tas_global_mean'].values
    recon_mean[experiment] = handle['recon_tas_mean'].values
    prior_gmt[experiment]  = handle['prior_tas_global_mean'].values
    proxy_metadata[experiment] = handle['proxy_metadata'].values
    handle.close()


#%% CALCULATIONS

# Find the indices for different periods
ind_0ka   = np.where((ages_da >= 0)     & (ages_da <= 1000))[0]
ind_6ka   = np.where((ages_da >= 5500)  & (ages_da <= 6500))[0]
ind_12ka  = np.where((ages_da >= 11000) & (ages_da <= 12000))[0]
ind_0_6ka = np.where((ages_da >= 0)     & (ages_da <= 6000))[0]

# Calculate NH and SH hemispheric-means
recon_nh,recon_sh,anom_6ka,trend_6ka = {},{},{},{}
for experiment in experiments:
    #
    print('Computing hemispheric means:',experiment)
    recon_nh[experiment] = utils_DA.spatial_mean(recon_mean[experiment],lat,lon,0,90,0,360,1,2)
    recon_sh[experiment] = utils_DA.spatial_mean(recon_mean[experiment],lat,lon,-90,0,0,360,1,2)
    #
    # Compute the relative mid-Holocene temperature for each reconstruction
    anom_6ka[experiment]  = np.mean(recon_gmt[experiment][ind_6ka,:]) - np.mean(recon_gmt[experiment][ind_0ka,:])
    #
    # Compute regression of temperatures from 6-0 ka
    recon_gm_mean = np.mean(recon_gmt[experiment],axis=1)
    trend_6ka[experiment],intercept,rvalue,pvalue,_ = stats.linregress(ages_da[ind_0_6ka],recon_gm_mean[ind_0_6ka])
    trend_6ka[experiment] = -1000*trend_6ka[experiment]

# If requested, remove the mean of the 0-1ka age interval from the reconstruction
if remove_0_1ka:
    indices_ref_recon = np.where((ages_da >= 0) & (ages_da <= 1000))[0]
    for experiment in experiments:
        recon_gmt[experiment] = recon_gmt[experiment] - np.mean(recon_gmt[experiment][indices_ref_recon,:])
        recon_nh[experiment]  = recon_nh[experiment]  - np.mean(recon_nh[experiment][indices_ref_recon])
        recon_sh[experiment]  = recon_sh[experiment]  - np.mean(recon_sh[experiment][indices_ref_recon])


#%% FIGURES
plt.style.use('ggplot')
experiments_to_plot = ['author_interp','annual','summer','winter']

# Plot time series of the reconstructions
f, ax1 = plt.subplots(1,1,figsize=(16,8))

spacing = ['','           ','         ','            ']
for i,experiment in enumerate(experiments_to_plot):
    ax1.fill_between(ages_da,np.nanpercentile(recon_gmt[experiment],2.5,axis=1),np.nanpercentile(recon_gmt[experiment],97.5,axis=1),color=colors[experiment],alpha=0.25)
    ax1.plot(ages_da,np.nanmean(recon_gmt[experiment],axis=1),color=colors[experiment],linewidth=3,label=experiment+spacing[i]+' (6 - 0.5 ka = '+str('{:.2f}'.format(anom_6ka[experiment]))+'$^\circ$C)')

ax1.axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
ax1.set_ylim(-2.6,.5)
ax1.set_xlim(12000,0)
ax1.set_xlabel('Age (yr BP)',fontsize=20)
ax1.set_ylabel('$\Delta$ Global-mean temperature ($^\circ$C)',fontsize=20)
ax1.legend(fontsize=20,loc='lower right')
ax1.tick_params(axis='both',which='major',labelsize=20)
ax1.set_title('Seasonality sensitivity experiments\nGlobal-mean temperatures (with 95% uncertainties)',loc='center',fontsize=24)

if save_instead_of_plot:
    plt.savefig('figures/PaperFig9_seasonality_gmt.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Plot time series of the NH and SH-means
f, ax = plt.subplots(len(experiments_to_plot)-1,1,figsize=(16,16),sharex=True,sharey=True)
ax = ax.ravel()

for i in range(len(experiments_to_plot)-1):
    experiment = experiments_to_plot[i+1]
    ax[i].plot(ages_da,recon_nh[experiment]-recon_nh['author_interp'],color='tab:red', linewidth=2,label='NH difference')
    ax[i].plot(ages_da,recon_sh[experiment]-recon_sh['author_interp'],color='tab:blue',linewidth=2,label='SH difference')
    ax[i].axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
    ax[i].legend(fontsize=12,loc='lower right',ncol=2)
    ax[i].set_ylabel('$\Delta$T ($^\circ$C)',fontsize=20)
    ax[i].tick_params(axis='both',which='major',labelsize=20)
    ax[i].set_title(experiment+' - author_interp',loc='left',fontsize=20)

ax[i].set_xlim(12000,0)
ax[i].set_xlabel('Age (yr BP)',fontsize=20)

plt.suptitle('Change in NH and SH-mean temperature\ncompared to the author_interp reconstruction',fontsize=20)

if save_instead_of_plot:
    plt.savefig('figures/seasonality_hemispheres.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Make a maps of changes
exp = 'author_interp'
def map_6ka_anom(exp,time_period):
    #
    # Calculate the temperature change between 6 and 0ka
    if time_period == 6:
        tas_anom_map = np.mean(recon_mean[exp][ind_6ka,:,:],axis=0) - np.mean(recon_mean[exp][ind_0ka,:,:],axis=0)
        tas_anom_map_prescibed = np.mean(recon_mean['author_interp'][ind_6ka,:,:],axis=0) - np.mean(recon_mean['author_interp'][ind_0ka,:,:],axis=0)
        values_range = np.linspace(-1,1,21)
    elif time_period == 12:
        tas_anom_map = np.mean(recon_mean[exp][ind_12ka,:,:],axis=0) - np.mean(recon_mean[exp][ind_0ka,:,:],axis=0)
        tas_anom_map_prescibed = np.mean(recon_mean['author_interp'][ind_12ka,:,:],axis=0) - np.mean(recon_mean['author_interp'][ind_0ka,:,:],axis=0)
        values_range = np.linspace(-5,5,21)
    #
    # Make a map
    plt.figure(figsize=(10,13))
    ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.Robinson()); ax1.set_global()
    tas_anom_map_cyclic,lon_cyclic = cutil.add_cyclic_point(tas_anom_map,coord=lon)
    map1 = ax1.contourf(lon_cyclic,lat,tas_anom_map_cyclic,values_range,extend='both',cmap='bwr',transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar1.ax.tick_params(labelsize=14)
    ax1.set_title('Temperature anomalies for '+str(time_period)+'-0 ka ($^\circ$C)\nfor experiment with all proxies set to '+exp+' seasonality',loc='center',fontsize=16)
    #
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    #
    if save_instead_of_plot:
        plt.savefig('figures/map_'+str(time_period)+'_0ka_alone_'+exp+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    #
    #
    if exp == 'author_interp': return
    #
    # Make a map
    plt.figure(figsize=(10,13))
    ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.Robinson()); ax1.set_global()
    tas_anom_map_cyclic,lon_cyclic = cutil.add_cyclic_point(tas_anom_map-tas_anom_map_prescibed,coord=lon)
    map1 = ax1.contourf(lon_cyclic,lat,tas_anom_map_cyclic,np.linspace(-1,1,21),extend='both',cmap='bwr',transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar1.ax.tick_params(labelsize=14)
    ax1.set_title('Temperature anomalies for '+str(time_period)+'-0 ka ($^\circ$C)\nfor experiment: '+exp+'-author_interp',loc='center',fontsize=16)
    #
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    #
    if save_instead_of_plot:
        plt.savefig('figures/map_'+str(time_period)+'_0ka_diff_'+exp+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

for exp in experiments_to_plot:
   map_6ka_anom(exp,6)
   map_6ka_anom(exp,12)

