#==============================================================================
# This computes trend maps in the Holocene reconstruction. This is Figs. 6 and
# 7 in the paper.
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
from scipy import stats
import seaborn as sns
import pandas as pd
from matplotlib import colors

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
gmt_recon_all  = handle['recon_tas_global_mean'].values
recon_mean     = handle['recon_tas_mean'].values
ages_da        = handle['ages'].values
lat            = handle['lat'].values
lon            = handle['lon'].values
proxy_values   = handle['proxy_values'].values
proxy_recon    = handle['proxyrecon_mean'].values
proxy_metadata = handle['proxy_metadata'].values
proxies_assimilated_all = handle['proxies_assimilated'].values
handle.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Calculate the mean GMT
gmt_recon_mean = np.mean(gmt_recon_all,axis=1)

# Figure out which proxies were assimilated
ind_ages_6_0ka  = np.where((ages_da >= 0)    & (ages_da <= 6000))[0]
ind_ages_12_6ka = np.where((ages_da >= 6000) & (ages_da <= 12000))[0]
ind_assim = np.where(np.mean(proxies_assimilated_all,axis=0) > 0)[0]
ind_assim_6_0ka  = np.where(np.mean(proxies_assimilated_all[ind_ages_6_0ka,:],axis=0)  > 0)[0]
ind_assim_12_6ka = np.where(np.mean(proxies_assimilated_all[ind_ages_12_6ka,:],axis=0) > 0)[0]

# Get proxy lats and lons
n_proxies = proxy_metadata.shape[0]
proxy_lats = proxy_metadata[:,2].astype(float)
proxy_lons = proxy_metadata[:,3].astype(float)
proxy_seasonality = proxy_metadata[:,5]

# Function to compute a regression over a given time range
age_max,age_min = 6000,0
def regression_calc(age_max,age_min):
    #
    ind_ages = np.where((ages_da >= age_min) & (ages_da <= age_max))[0]
    n_lat = len(lat)
    n_lon = len(lon)
    slope     = np.zeros((n_lat,n_lon)); slope[:]     = np.nan
    intercept = np.zeros((n_lat,n_lon)); intercept[:] = np.nan
    for j in range(n_lat):
        for i in range(n_lon):
            slope[j,i],intercept[j,i],_,_,_ = stats.linregress(-1*ages_da[ind_ages]/1000,recon_mean[ind_ages,j,i])
    #
    # Complute the global-mean regression
    slope_gmt,intercept_gmt,_,_,_ = stats.linregress(-1*ages_da[ind_ages]/1000,gmt_recon_mean[ind_ages])
    #
    return slope,intercept,slope_gmt,intercept_gmt

# Calculate the regressions for the proxies or proxy reconstructions
ts_all,age_max,age_min = proxy_values,12000,6000
def proxy_regression_calc(ts_all,age_max,age_min):
    #
    ind_ages = np.where((ages_da >= age_min) & (ages_da <= age_max))[0]
    ages_selected = ages_da[ind_ages]
    #
    slope     = np.zeros((n_proxies)); slope[:]     = np.nan
    intercept = np.zeros((n_proxies)); intercept[:] = np.nan
    n_points  = np.zeros((n_proxies)); n_points[:]  = np.nan
    for i in range(n_proxies):
        ts_selected = ts_all[ind_ages,i]
        ind_valid = np.isfinite(ts_selected)
        n_points[i] = sum(ind_valid)
        # Only compute regressions if at least half of the ages have data.
        if sum(ind_valid) >= len(ind_valid)/2:
        #if sum(ind_valid) >= 100:  # This is the older version.
            slope[i],intercept[i],_,_,_ = stats.linregress(-1*ages_selected[ind_valid]/1000,ts_selected[ind_valid])
    #
    return slope,intercept


#%%
# Calculate the regressions everywhere
trend_12_6ka,_,trend_gmt_12_6ka,_ = regression_calc(12000,6000)
trend_6_0ka,_, trend_gmt_6_0ka,_  = regression_calc(6000,0)
proxy_trend_12_6ka,_ = proxy_regression_calc(proxy_values,12000,6000)
proxy_trend_6_0ka,_  = proxy_regression_calc(proxy_values,6000,0)
proxy_recon_12_6ka,_ = proxy_regression_calc(proxy_recon,12000,6000)
proxy_recon_6_0ka,_  = proxy_regression_calc(proxy_recon,6000,0)

#%%
# Plot proxy anomalies for this location
def plot_proxies(values_selected,lats_selected,lons_selected,seasonality_selected,anomaly_value,ax,legend_loc,colors_discrete):
    for i in range(len(values_selected)):
        if   (seasonality_selected[i] == 'annual'):                                              marker_symbol = 'o'
        elif (seasonality_selected[i] == 'summerOnly') | (seasonality_selected[i] == 'summer+'): marker_symbol = '^'
        elif (seasonality_selected[i] == 'winterOnly') | (seasonality_selected[i] == 'winter+'): marker_symbol = 'v'
        marker_size = 100
        if np.isnan(values_selected[i]): continue
        ax.scatter([lons_selected[i]],[lats_selected[i]],marker_size,c=[values_selected[i]],marker=marker_symbol,edgecolor='k',alpha=1,cmap='bwr',norm=colors_discrete,linewidths=1,transform=ccrs.PlateCarree())
    #
    # Add a legend
    legend = [mlines.Line2D([],[],markeredgecolor='k',markerfacecolor='w',marker='o',markersize=10,linewidth=0,label='Annual'),
              mlines.Line2D([],[],markeredgecolor='k',markerfacecolor='w',marker='^',markersize=10,linewidth=0,label='Summer'),
              mlines.Line2D([],[],markeredgecolor='k',markerfacecolor='w',marker='v',markersize=10,linewidth=0,label='Winter')]
    #ax.legend(handles=legend,loc=legend_loc,fontsize=14,ncol=1)


#%% FIGURES
plt.style.use('ggplot')

# Make a maps of changes
#region,proxy_type,file_txt = 'NAmerica_and_Europe','proxyvalues','PaperFigS4'
def make_trend_map(region,proxy_type,file_txt):
    if region == 'global':
        plt.figure(figsize=(22,13))
        ax1 = plt.subplot2grid((2,1),(0,0),projection=ccrs.Robinson()); ax1.set_global()
        ax2 = plt.subplot2grid((2,1),(1,0),projection=ccrs.Robinson()); ax2.set_global()
        colorbar_fraction = 0.08
        legend_loc = 'lower left'
    elif region == 'NAmerica_and_Europe':
        plt.figure(figsize=(20,14))
        ax1 = plt.subplot2grid((2,1),(0,0),projection=ccrs.PlateCarree()); ax1.set_extent([-140,40,25,85],crs=ccrs.PlateCarree())
        ax2 = plt.subplot2grid((2,1),(1,0),projection=ccrs.PlateCarree()); ax2.set_extent([-140,40,25,85],crs=ccrs.PlateCarree())
        colorbar_fraction = 0.115
        legend_loc = 'lower left'
    #
    # Create the same colorbar for both contourf and scatter
    range_selected1 = np.linspace(-1,1,21)
    range_selected2 = np.linspace(-.2,.2,21)
    colors_discrete1 = colors.BoundaryNorm(range_selected1,plt.cm.bwr.N)
    colors_discrete2 = colors.BoundaryNorm(range_selected2,plt.cm.bwr.N)
    #
    trend_12_6ka_cyclic,lon_cyclic = cutil.add_cyclic_point(trend_12_6ka,coord=lon)
    map1 = ax1.contourf(lon_cyclic,lat,trend_12_6ka_cyclic,range_selected1,extend='both',norm=colors_discrete1,cmap='bwr',transform=ccrs.PlateCarree())
    if proxy_type == 'proxyvalues': plot_proxies(proxy_trend_12_6ka[ind_assim_12_6ka],proxy_lats[ind_assim_12_6ka],proxy_lons[ind_assim_12_6ka],proxy_seasonality[ind_assim_12_6ka],1,ax1,legend_loc,colors_discrete1)
    else: ax1.scatter(proxy_lons[ind_assim_12_6ka],proxy_lats[ind_assim_12_6ka],5,c='k',marker='o',alpha=1,transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=colorbar_fraction,pad=0.02)
    colorbar1.ax.tick_params(labelsize=14)
    colorbar1.ax.set_facecolor('none')
    ax1.set_title('(a) 12 to 6 ka',loc='center',fontsize=20)
    #
    trend_6_0ka_cyclic,lon_cyclic = cutil.add_cyclic_point(trend_6_0ka,coord=lon)
    map2 = ax2.contourf(lon_cyclic,lat,trend_6_0ka_cyclic,range_selected2,extend='both',norm=colors_discrete2,cmap='bwr',transform=ccrs.PlateCarree())
    if proxy_type == 'proxyvalues': plot_proxies(proxy_trend_6_0ka[ind_assim_6_0ka],proxy_lats[ind_assim_6_0ka],proxy_lons[ind_assim_6_0ka],proxy_seasonality[ind_assim_6_0ka],.2,ax2,legend_loc,colors_discrete2)
    else: ax2.scatter(proxy_lons[ind_assim_6_0ka],proxy_lats[ind_assim_6_0ka],5,c='k',marker='o',alpha=1,transform=ccrs.PlateCarree())
    colorbar2 = plt.colorbar(map2,orientation='horizontal',ax=ax2,fraction=colorbar_fraction,pad=0.02)
    colorbar2.set_label('Temperature trends ($^\circ$C / kyr)',fontsize=16)
    colorbar2.ax.tick_params(labelsize=14)
    colorbar2.ax.set_facecolor('none')
    ax2.set_title('(b) 6 to 0 ka',loc='center',fontsize=20)
    #
    for ax in [ax1,ax2]:
        ax.coastlines()
        ax.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
        ax.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
        ax.spines['geo'].set_edgecolor('black')
    #
    if save_instead_of_plot:
        plt.savefig('figures/Paper'+file_txt+'_trend_maps_'+region+'_'+proxy_type+'_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

make_trend_map('global','basicproxies','Fig6')
make_trend_map('NAmerica_and_Europe','proxyvalues','Fig7')


#%%
# Make histograms of changes
print(np.nanmin(proxy_trend_12_6ka),np.nanmax(proxy_trend_12_6ka))
print(np.nanmin(proxy_recon_12_6ka),np.nanmax(proxy_recon_12_6ka))

print(np.nanmin(proxy_trend_6_0ka),np.nanmax(proxy_trend_6_0ka))
print(np.nanmin(proxy_recon_6_0ka),np.nanmax(proxy_recon_6_0ka))


# Plot histograms of values
f, ax = plt.subplots(2,1,figsize=(12,10),sharex=False,sharey=False)
ax = ax.ravel()
bins_values_12_6ka = np.arange(-2,2,.1)
bins_values_6_0ka  = np.arange(-2,2,.1)
#TOOD: Consider changing the bounds above

ax[0].hist(proxy_trend_12_6ka,bins=bins_values_12_6ka,color='tab:red', alpha=0.5,label='Proxies;                        median: '+str('{:.2f}'.format(np.nanmedian(proxy_trend_12_6ka)))+'$^\circ$C / kyr')
ax[0].hist(proxy_recon_12_6ka,bins=bins_values_12_6ka,color='tab:blue',alpha=0.5,label='Proxy reconstructions; median: '+str('{:.2f}'.format(np.nanmedian(proxy_recon_12_6ka)))+'$^\circ$C / kyr')
ax[0].axvline(x=np.nanmedian(proxy_trend_12_6ka),color='tab:red', alpha=1,linestyle=':',linewidth=2)
ax[0].axvline(x=np.nanmedian(proxy_recon_12_6ka),color='tab:blue',alpha=1,linestyle=':',linewidth=2)
ax[0].axvline(x=trend_gmt_12_6ka,                color='k',       alpha=1,linestyle=':',linewidth=2)
ax[0].set_xlim(-2,2)
ax[0].set_title('(a) 12 to 6 ka',loc='left',fontsize=16)
ax[0].set_xlabel('Temperature trend ($^\circ$C / kyr)',fontsize=16)
ax[0].legend(loc=1)

ax[1].hist(proxy_trend_6_0ka,bins=bins_values_6_0ka,color='tab:red', alpha=0.5,label='Proxies;                        median: '+str('{:.2f}'.format(np.nanmedian(proxy_trend_6_0ka)))+'$^\circ$C / kyr')
ax[1].hist(proxy_recon_6_0ka,bins=bins_values_6_0ka,color='tab:blue',alpha=0.5,label='Proxy reconstructions; median: '+str('{:.2f}'.format(np.nanmedian(proxy_recon_6_0ka)))+'$^\circ$C / kyr')
ax[1].axvline(x=np.nanmedian(proxy_trend_6_0ka),color='tab:red', alpha=1,linestyle=':',linewidth=2)
ax[1].axvline(x=np.nanmedian(proxy_recon_6_0ka),color='tab:blue',alpha=1,linestyle=':',linewidth=2)
ax[1].axvline(x=trend_gmt_6_0ka,                color='k',       alpha=1,linestyle=':',linewidth=2)
ax[1].set_xlim(-2,2)
ax[1].set_title('(b) 6 to 0 ka',loc='left',fontsize=16)
ax[1].set_xlabel('Temperature trend ($^\circ$C / kyr)',fontsize=16)
ax[1].legend(loc=1)

for i in range(2):
    ax[i].set_ylabel('Frequency',fontsize=16)
    ax[i].tick_params(axis='both',which='major',labelsize=14)
    ax[i].axvline(x=0,color='k',linestyle='--')

plt.suptitle('Temperature trends over different periods',fontsize=20)
f.tight_layout()
f.subplots_adjust(top=.9)

if save_instead_of_plot:
    plt.savefig('figures/trend_hists_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Calculate trends over 2k periods
trends_2k            = {}
trends_2k_gmt        = {}
proxy_trend_selected = {}
proxy_recon_selected = {}
trends_2k['temp']   = []
trends_2k['type']   = []
trends_2k['period'] = []
trends_2k_gmt_array = []
n_age_span = 2000
for i,age_begin in enumerate(np.arange(12000,0,-n_age_span)):
    age_end = age_begin - n_age_span
    key = str(int(age_begin/1000))+'-'+str(int(age_end/1000))+' ka'
    print(key)
    #
    _,_,trends_2k_gmt[key],_ = regression_calc(age_begin,age_end)
    proxy_trend_selected[key],_ = proxy_regression_calc(proxy_values,age_begin,age_end)
    proxy_recon_selected[key],_ = proxy_regression_calc(proxy_recon, age_begin,age_end)
    trends_2k_selected = np.append(proxy_trend_selected[key],proxy_recon_selected[key])
    #
    trends_2k_gmt_array.append(trends_2k_gmt[key])
    types = np.append(['proxy']*n_proxies,['recon']*n_proxies)
    trends_2k['temp'].extend(trends_2k_selected)
    trends_2k['type'].extend(types)
    trends_2k['period'].extend([key]*n_proxies*2)

trends_2k_df = pd.DataFrame(trends_2k)


#%%
# Calculate the regressions for 2ka segments
f, ax = plt.subplots(6,1,figsize=(10,20),sharex=False,sharey=False)
ax = ax.ravel()
#bins_values = np.arange(-2,2,.1)
bins_values = np.arange(-1,1,.05)

n_age_span = 2000
letters = ['a','b','c','d','e','f']
for i,age_begin in enumerate(np.arange(12000,0,-n_age_span)):
    age_end = age_begin - n_age_span
    key = str(int(age_begin/1000))+'-'+str(int(age_end/1000))+' ka'
    print('===')
    print(key)
    print('Proxy values:',np.nanmedian(proxy_trend_selected[key]))
    print('Proxy recon: ',np.nanmedian(proxy_recon_selected[key]))
    print('GMT:         ',trends_2k_gmt[key])
    ax[i].hist(proxy_trend_selected[key],bins=bins_values,color='tab:red', alpha=0.5,label='Proxies;          median: '+str('{:.2f}'.format(np.nanmedian(proxy_trend_selected[key])))+'$^\circ$C / kyr')
    ax[i].hist(proxy_recon_selected[key],bins=bins_values,color='tab:blue',alpha=0.5,label='Proxy recons; median: '+str('{:.2f}'.format(np.nanmedian(proxy_recon_selected[key])))+'$^\circ$C / kyr')
    ax[i].axvline(x=np.nanmedian(proxy_trend_selected[key]),color='tab:red', alpha=1,linestyle=':',linewidth=2)
    ax[i].axvline(x=np.nanmedian(proxy_recon_selected[key]),color='tab:blue',alpha=1,linestyle=':',linewidth=2)
    ax[i].axvline(x=trends_2k_gmt[key],                     color='k',       alpha=1,linestyle=':',linewidth=2,label='Recon, global mean:   '+str('{:.2f}'.format(trends_2k_gmt[key]))+'$^\circ$C / kyr')
    #ax[i].set_xlim(-2,2)
    ax[i].set_xlim(-1,1)
    ax[i].set_title('('+letters[i]+') '+str(int(age_begin/1000))+' to '+str(int(age_end/1000))+' ka',loc='left',fontsize=16)
    ax[i].set_xlabel('Temperature trend ($^\circ$C / kyr)',fontsize=16)
    ax[i].legend(loc='upper left')
    ax[i].set_ylabel('Frequency',fontsize=16)
    ax[i].tick_params(axis='both',which='major',labelsize=14)
    ax[i].axvline(x=0,color='k',linestyle='--')

plt.suptitle('Temperature trends over different periods',fontsize=20)
f.tight_layout()
f.subplots_adjust(top=.9)

if save_instead_of_plot:
    plt.savefig('figures/trend_hists_2k_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Calculate the regressions for 2ka segments
f, ax1 = plt.subplots(1,1,figsize=(12,7))

sns.boxplot(data=trends_2k_df,y='temp',x='period',hue='type',showfliers=False,ax=ax1)
ax1.scatter(np.arange(6),trends_2k_gmt_array,100,c='k')
ax1.axhline(y=0,color='k',alpha=1,linestyle=':',linewidth=1)
#ax1.set_ylim(-1,1)
ax1.set_ylim(-2,3)
ax1.set_ylabel('Temperature trend ($^\circ$C / kyr)',fontsize=16)
ax1.set_xlabel('Period',fontsize=16)
ax1.set_title('Temperature trends over different periods',loc='center',fontsize=20)

if save_instead_of_plot:
    plt.savefig('figures/trend_boxplots_2k_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()



