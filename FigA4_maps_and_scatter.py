#==============================================================================
# This script compares proxies to reconstructed proxies in several different
# ways. This makes Fig. A4 in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
import xarray as xr
import copy
from scipy import stats

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
recon_mean         = handle['recon_tas_mean'].values
recon_mean_proxies = handle['proxyrecon_mean'].values
proxy_values       = handle['proxy_values'].values
proxy_metadata     = handle['proxy_metadata'].values
proxies_assimilated_all = handle['proxies_assimilated'].values
ages_da            = handle['ages'].values
lat                = handle['lat'].values
lon                = handle['lon'].values
options_da         = handle['options'].values
handle.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Get proxy lats and lons
proxy_lats = proxy_metadata[:,2].astype(float)
proxy_lons = proxy_metadata[:,3].astype(float)
proxy_lons[proxy_lons < 0] = proxy_lons[proxy_lons < 0] + 360

# Figure out which proxies were assimilated
ind_proxies_assimilated = np.where(np.mean(proxies_assimilated_all,axis=0) > 0)[0]

# Count data coverage
proxy_values_assim = copy.deepcopy(proxy_values)
proxy_values_assim[proxies_assimilated_all == 0] = np.nan

# Calculate the correlations and slopes of decadal values through time
n_ages = len(ages_da)
corr_through_time     = np.zeros((n_ages)); corr_through_time[:]     = np.nan
slope_through_time    = np.zeros((n_ages)); slope_through_time[:]    = np.nan
nproxies_through_time = np.zeros((n_ages)); nproxies_through_time[:] = np.nan
for i in range(n_ages):
    #
    # Get the proxy and reconstructed proxy values
    proxy_recon_for_age  = recon_mean_proxies[i,:]
    proxy_values_for_age = proxy_values_assim[i,:]
    ind_valid = np.isfinite(proxy_values_for_age) & np.isfinite(proxy_recon_for_age)
    slope_through_time[i],_,corr_through_time[i],_,_ = stats.linregress(proxy_values_for_age[ind_valid],proxy_recon_for_age[ind_valid])
    nproxies_through_time[i] = sum(ind_valid)


#%% FIGURES
plt.style.use('ggplot')

# Function to make a map of a particular age in the reconstruction and the assimilated proxies
ages_anom,ages_ref = [6000,6010],[3000,5000]
def map_recon_and_proxies(ages_anom,ages_ref):
    #
    # Compute the anomalies
    indices_anom = np.where((ages_da >= ages_anom[0]) & (ages_da <= ages_anom[1]))[0]
    indices_ref  = np.where((ages_da >= ages_ref[0])  & (ages_da <= ages_ref[1]))[0]
    if (ages_ref[0] == 3000) & (ages_ref[1] == 5000): 
        recon_mean_for_age       = np.nanmean(recon_mean[indices_anom,:,:],axis=0)
        proxy_recon_for_age_all  = np.nanmean(recon_mean_proxies[indices_anom,:],axis=0)
        proxy_values_for_age_all = np.nanmean(proxy_values_assim[indices_anom,:],axis=0)
    else:
        recon_mean_for_age       = np.nanmean(recon_mean[indices_anom,:,:],axis=0)       - np.nanmean(recon_mean[indices_ref,:,:],axis=0)
        proxy_recon_for_age_all  = np.nanmean(recon_mean_proxies[indices_anom,:],axis=0) - np.nanmean(recon_mean_proxies[indices_ref,:],axis=0)
        proxy_values_for_age_all = np.nanmean(proxy_values_assim[indices_anom,:],axis=0) - np.nanmean(proxy_values_assim[indices_ref,:],axis=0)
    #
    # Get data for assimilated proxy
    proxy_recon_for_age  = proxy_recon_for_age_all[ind_proxies_assimilated]
    proxy_values_for_age = proxy_values_for_age_all[ind_proxies_assimilated]
    proxy_lats           = proxy_metadata[ind_proxies_assimilated,2].astype(float)
    proxy_lons           = proxy_metadata[ind_proxies_assimilated,3].astype(float)
    proxy_seasonality    = proxy_metadata[ind_proxies_assimilated,5]
    #
    #
    # Make a map of changes
    plt.figure(figsize=(18,18))
    ax1 = plt.subplot2grid((5,5),(0,0),rowspan=2,colspan=3,projection=ccrs.Robinson()); ax1.set_global()
    anomaly_value = 2
    recon_mean_for_age_cyclic,lon_cyclic = cutil.add_cyclic_point(recon_mean_for_age,coord=lon)
    map1 = ax1.contourf(lon_cyclic,lat,recon_mean_for_age_cyclic,np.linspace(-1*anomaly_value,anomaly_value,21),extend='both',cmap='bwr',vmin=-1*anomaly_value,vmax=anomaly_value,transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    colorbar = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar.set_label('Temperature',fontsize=14)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.ax.set_facecolor('none')
    #
    # Plot proxy anomalies for this location
    for i in range(len(ind_proxies_assimilated)):
        if   (proxy_seasonality[i] == 'annual'):                                           marker_symbol = 'o'
        elif (proxy_seasonality[i] == 'summerOnly') | (proxy_seasonality[i] == 'summer+'): marker_symbol = '^'
        elif (proxy_seasonality[i] == 'winterOnly') | (proxy_seasonality[i] == 'winter+'): marker_symbol = 'v'
        if np.isnan(proxy_values_for_age[i]): continue
        marker_size = 50
        ax1.scatter([proxy_lons[i]],[proxy_lats[i]],marker_size,c=[proxy_values_for_age[i]],marker=marker_symbol,edgecolor='k',alpha=1,cmap='bwr',vmin=-1*anomaly_value,vmax=anomaly_value,linewidths=1,transform=ccrs.PlateCarree())
    #
    # Add a legend
    legend = [mlines.Line2D([],[],markeredgecolor='k',markerfacecolor='w',marker='o',markersize=10,linewidth=0,label='Annual'),
              mlines.Line2D([],[],markeredgecolor='k',markerfacecolor='w',marker='^',markersize=10,linewidth=0,label='Summer'),
              mlines.Line2D([],[],markeredgecolor='k',markerfacecolor='w',marker='v',markersize=10,linewidth=0,label='Winter')]
    ax1.legend(handles=legend,loc='lower left',fontsize=14,ncol=1)
    ax1.set_title('(a) Maps for reconstruction and proxy records ($^\circ$C)',fontsize=18)
    #
    # Make a scatterplot
    ax2 = plt.subplot2grid((5,5),(0,3),rowspan=2,colspan=2)
    range_min = np.floor(min([np.nanmin(proxy_values_for_age),np.nanmin(proxy_recon_for_age)]))
    range_max = np.ceil(max([np.nanmax(proxy_values_for_age),np.nanmax(proxy_recon_for_age)]))
    ind_valid = np.isfinite(proxy_values_for_age) & np.isfinite(proxy_recon_for_age)
    r2 = np.square(np.corrcoef(proxy_values_for_age[ind_valid],proxy_recon_for_age[ind_valid])[0,1])
    ax2.scatter(proxy_values_for_age[ind_valid],proxy_recon_for_age[ind_valid],20)
    ax2.scatter(np.mean(proxy_values_for_age[ind_valid]),np.mean(proxy_recon_for_age[ind_valid]),100,c='k')
    ax2.plot([range_min,range_max],[range_min,range_max],color='gray',linestyle=':')
    ax2.set_xlim(range_min,range_max)
    ax2.set_ylim(range_min,range_max)
    ax2.set_xlabel('Proxy record $\Delta$T ($^\circ$C)',fontsize=16)
    ax2.set_ylabel('Reconstructed record $\Delta$T ($^\circ$C)',fontsize=16)
    ax2.set_title('(b) Scatterplot, $R^2$='+str('{:.2f}'.format(r2)),fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #
    ax3 = plt.subplot2grid((5,5),(2,0),colspan=5)
    ax3.plot(ages_da,np.nanmean(proxy_values_assim,axis=1),'tab:blue',label='Proxy mean')
    ax3.plot(ages_da,np.nanmean(recon_mean_proxies,axis=1),'tab:red', label='Reconstructed proxy mean')
    ax3.legend(fontsize=14)
    ax3.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=16)
    ax3.set_title('(c) Means of proxy records and reconstructed records through time',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #
    ax4 = plt.subplot2grid((5,5),(3,0),colspan=5)
    ax4.plot(ages_da,slope_through_time,'k',label='Proxy mean')
    ax4.set_ylabel('Slope',fontsize=16)
    ax4.set_title('(d) Slope between proxy records and reconstructed records through time',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #
    ax5 = plt.subplot2grid((5,5),(4,0),colspan=5)
    ax5.plot(ages_da,corr_through_time,'k',label='Proxy mean')
    ax5.set_ylabel('Correlation',fontsize=16)
    ax5.set_title('(e) Correlation of proxy records and reconstructed records through time',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #
    for ax in [ax3,ax4,ax5]:
        ax.axvspan(ages_anom[0],ages_anom[1],facecolor='gray',alpha=0.5)
        ax.set_xlim(ages_da[-1],ages_da[0])
        ax.set_xlabel('Age (B.P.)',fontsize=16)
    #
    plt.tight_layout()
    if save_instead_of_plot:
        plt.savefig('figures/PaperFigA4_age_'+str(int(ages_anom[0])).zfill(5)+'_'+str(int(ages_anom[1])).zfill(5)+'_vs_'+str(int(ages_ref[0])).zfill(5)+'_'+str(int(ages_ref[1])).zfill(5)+'_yr_BP_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

timestep = np.mean(ages_da[1:]-ages_da[:-1])
map_recon_and_proxies([6000,6000+timestep],[3000,5000])

"""
for i in np.arange(0,12000,1000):
    map_recon_and_proxies([i,i+timestep],[3000,5000])

map_recon_and_proxies([12000-timestep,12000],[3000,5000])
"""
