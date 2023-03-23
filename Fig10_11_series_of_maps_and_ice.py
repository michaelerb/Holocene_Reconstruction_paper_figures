#==============================================================================
# Make a series of maps of ice sheets and reconstructed temperature. This is
# Figs. 10 and 11 in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
import xarray as xr
import copy
from matplotlib import colors

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
gmt_recon_all   = handle['recon_tas_global_mean'].values
recon_mean      = handle['recon_tas_mean'].values
ages_da         = handle['ages'].values
lat             = handle['lat'].values
lon             = handle['lon'].values
proxy_values    = handle['proxy_values'].values
proxy_recon     = handle['proxyrecon_mean'].values
proxy_metadata  = handle['proxy_metadata'].values
proxy_assim_all = handle['proxies_assimilated'].values
handle.close()

# Ice thickness data
data_dir = '/projects/pd_lab/data/climate_forcings/'
handle = xr.open_dataset(data_dir+'deglaciation_pmip4/ICE-6G-C/icethick/ICE-6G_C_IceThickness_1deg.nc',decode_times=False)
ice_thickness = handle['stgit'].values
ages_ice      = handle['Time'].values*1000
lat_ice       = handle['Lat'].values
lon_ice       = handle['Lon'].values
handle.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Get values of assimilated proxies
proxy_values_assim = copy.deepcopy(proxy_values)
proxy_values_assim[proxy_assim_all == 0] = np.nan

# Figure out which proxies were assimilated
ind_proxies_assimilated = np.where(np.mean(proxy_assim_all,axis=0) > 0)[0]
ind_proxies_leftout     = np.where(np.mean(proxy_assim_all,axis=0) == 0)[0]

# Get proxy lats and lons
proxy_lats = proxy_metadata[:,2].astype(float)
proxy_lons = proxy_metadata[:,3].astype(float)
proxy_season = proxy_metadata[:,5]
ind_annual = proxy_season == 'annual'

# Compute the verticies of the grid, for plotting purposes
lon_wrap = copy.deepcopy(lon)
lon_wrap = np.insert(lon_wrap,0,lon_wrap[-1]-360)  # Add the right-most lon point to the left
lon_wrap = np.append(lon_wrap,lon_wrap[1]+360)     # Add the left-most lon point to the right
lon_edges = (lon_wrap[:-1] + lon_wrap[1:])/2

lat_edges = copy.deepcopy(lat)
lat_edges = (lat[:-1] + lat[1:])/2
lat_edges = np.insert(lat_edges,0,-90)  # Add the South Pole to the beginning
lat_edges = np.append(lat_edges,90)     # Add the North Pole to the end


#%% FUNCTIONS

# A function to grid proxy values to the nearest DA gridpoint
#values_selected = proxy_values_assim
def grid_values(values_selected,lats_selected,lons_selected):
    #
    proxy_lons_adj = copy.deepcopy(proxy_lons)
    proxy_lons_adj[proxy_lons_adj > lon_edges[-1]] = proxy_lons_adj[proxy_lons_adj > lon_edges[-1]] - 360
    #
    n_time    = values_selected.shape[0]
    n_latlon = len(lat)*len(lon)
    values_gridded = np.zeros((n_time,n_latlon)); values_gridded[:] = np.nan
    lats_gridded   = np.zeros((n_latlon));        lats_gridded[:]   = np.nan
    lons_gridded   = np.zeros((n_latlon));        lons_gridded[:]   = np.nan
    counter = 0
    for j,lat_center in enumerate(lat):
        for i,lon_center in enumerate(lon):
            ind_in_range = np.where((lats_selected >= lat_edges[j]) & (lats_selected < lat_edges[j+1]) & (lons_selected >= lon_edges[i]) & (lons_selected < lon_edges[i+1]))[0]
            if len(ind_in_range) == 0: values_in_range = np.nan
            else: values_in_range = np.nanmean(values_selected[:,ind_in_range],axis=1)
            #print(j,i,lat_center,lon_center,len(ind_in_range))
            values_gridded[:,counter] = values_in_range
            lats_gridded[counter] = lat_center
            lons_gridded[counter] = lon_center
            counter += 1
    #
    ind_valid = np.where(np.sum(np.isfinite(values_gridded),axis=0) > 0)[0]
    values_gridded = values_gridded[:,ind_valid]
    lats_gridded   = lats_gridded[ind_valid]
    lons_gridded   = lons_gridded[ind_valid]
    #
    return values_gridded,lats_gridded,lons_gridded

values_gridded_all,lats_gridded_all,lons_gridded_all = grid_values(proxy_values_assim,proxy_lats,proxy_lons)
values_gridded_ann,lats_gridded_ann,lons_gridded_ann = grid_values(proxy_values_assim[:,ind_annual],proxy_lats[ind_annual],proxy_lons[ind_annual])


#%% FIGURES
plt.style.use('ggplot')

# Make a map of a series of ages for the reconstruction and the assimilated proxies
region = 'NAmerica'; show_proxies='values_gridded_ann'
def multi_map(region,filename_txt,show_proxies='circles'):
    #
    #ages_ref = [0,1000]
    ages_ref = [3000,5000]
    anomaly_value = 2
    #
    if   region == 'NAmerica': region_bounds = [-160,-45,25,75]; height = 11.5; top_adjust = 0.96
    elif region == 'Europe':   region_bounds = [-45,40,30,80];   height = 14;   top_adjust = 0.97
    elif region == 'India':    region_bounds = [60,150,0,50];    height = 15;   top_adjust = 0.97
    elif region == 'NAfrica':  region_bounds = [-30,60,0,40];    height = 13;   top_adjust = 0.97
    #
    plt.figure(figsize=(12,height))
    ax = {}
    for panel,age_old in enumerate(np.arange(12000,0,-1000)):
        #
        ages_anom = [age_old-1000,age_old]
        print(ages_anom)
        #
        # Compute the anomalies
        indices_anom = np.where((ages_da >= ages_anom[0]) & (ages_da <= ages_anom[1]))[0]
        indices_ref  = np.where((ages_da >= ages_ref[0])  & (ages_da <= ages_ref[1]))[0]
        recon_mean_for_age = np.nanmean(recon_mean[indices_anom,:,:],axis=0) - np.nanmean(recon_mean[indices_ref,:,:],axis=0)
        #
        indices_anom_ice = np.where((ages_ice >= ages_anom[0]) & (ages_ice <= ages_anom[1]))[0]
        indices_ref_ice  = np.where((ages_ice >= ages_ref[0])  & (ages_ice <= ages_ref[1]))[0]
        ice_for_age = np.nanmean(ice_thickness[indices_anom_ice,:,:],axis=0) - np.nanmean(ice_thickness[indices_ref_ice,:,:],axis=0)
        #
        # Make a map of changes
        xloc = int(panel/3)
        yloc = np.remainder(panel,3)
        ax[panel] = plt.subplot2grid((4,3),(xloc,yloc),projection=ccrs.PlateCarree())
        ax[panel].set_extent(region_bounds)
        #
        # Create the same colorbar for both contourf and scatter
        range_selected = np.linspace(-anomaly_value,anomaly_value,21)
        colors_discrete = colors.BoundaryNorm(range_selected,plt.cm.bwr.N)
        #
        recon_mean_for_age_cyclic,lon_cyclic = cutil.add_cyclic_point(recon_mean_for_age,coord=lon)
        map1 = ax[panel].contourf(lon_cyclic,lat,recon_mean_for_age_cyclic,range_selected,extend='both',norm=colors_discrete,cmap='bwr',transform=ccrs.PlateCarree())
        ax[panel].contour(lon_ice,lat_ice,ice_for_age,np.linspace(-5000,5000,21),colors='k',linewidth=2,transform=ccrs.PlateCarree())
        ax[panel].contour(lon_ice,lat_ice,ice_for_age,[0],                       colors='k',linewidth=4,transform=ccrs.PlateCarree())
        ax[panel].coastlines()
        ax[panel].add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
        ax[panel].gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
        ax[panel].spines['geo'].set_edgecolor('black')
        ax[panel].set_title(str(int(ages_anom[0]/1000))+'-'+str(int(ages_anom[1]/1000))+' ka',fontsize=18,loc='left')
        #
        proxy_values_for_age = np.nanmean(proxy_values_assim[indices_anom,:],axis=0) - np.nanmean(proxy_values_assim[indices_ref,:],axis=0)
        ind_valid = np.isfinite(proxy_values_for_age)
        #
        if show_proxies == 'circles':
            ax[panel].scatter(proxy_lons[ind_proxies_assimilated],proxy_lats[ind_proxies_assimilated],30,facecolors='none',marker='o',edgecolor='k',alpha=1,linewidths=.2,transform=ccrs.PlateCarree())
        elif show_proxies == 'values':
            ax[panel].scatter(proxy_lons[ind_valid],proxy_lats[ind_valid],30,c=proxy_values_for_age[ind_valid],marker='o',edgecolor='k',alpha=1,cmap='bwr',norm=colors_discrete,linewidths=.5,transform=ccrs.PlateCarree())
        elif show_proxies == 'values_annual':
            ax[panel].scatter(proxy_lons[ind_valid & ind_annual],proxy_lats[ind_valid & ind_annual],30,c=proxy_values_for_age[ind_valid & ind_annual],marker='o',edgecolor='k',alpha=1,cmap='bwr',norm=colors_discrete,linewidths=.5,transform=ccrs.PlateCarree())
        elif show_proxies == 'values_gridded':
            values_gridded_for_age = np.nanmean(values_gridded_all[indices_anom,:],axis=0) - np.nanmean(values_gridded_all[indices_ref,:],axis=0)
            ind_valid_grid = np.isfinite(values_gridded_for_age)
            ax[panel].scatter(lons_gridded_all[ind_valid_grid],lats_gridded_all[ind_valid_grid],30,c=values_gridded_for_age[ind_valid_grid],marker='o',edgecolor='k',alpha=1,cmap='bwr',norm=colors_discrete,linewidths=.5,transform=ccrs.PlateCarree())
        elif show_proxies == 'values_gridded_ann':
            values_gridded_for_age = np.nanmean(values_gridded_ann[indices_anom,:],axis=0) - np.nanmean(values_gridded_ann[indices_ref,:],axis=0)
            ind_valid_grid = np.isfinite(values_gridded_for_age)
            ax[panel].scatter(lons_gridded_ann[ind_valid_grid],lats_gridded_ann[ind_valid_grid],30,c=values_gridded_for_age[ind_valid_grid],marker='o',edgecolor='k',alpha=1,cmap='bwr',norm=colors_discrete,linewidths=.5,transform=ccrs.PlateCarree())
    #
    plt.tight_layout()
    plt.subplots_adjust(bottom=.1)
    plt.subplots_adjust(top=top_adjust)
    axes_all = [ax[0],ax[1],ax[2],ax[3],ax[4],ax[5],ax[6],ax[7],ax[8],ax[9],ax[10],ax[11]]
    colorbar = plt.colorbar(map1,orientation='horizontal',ax=axes_all,fraction=0.08,aspect=40,pad=0.02)
    colorbar.set_label('$\Delta$Temperature ($^\circ$C)',fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    colorbar.ax.set_facecolor('none')
    plt.suptitle('Temperature and ice anomalies vs. '+str(int(ages_ref[0]/1000))+'-'+str(int(ages_ref[1]/1000))+' kyr BP',fontsize=20)
    if save_instead_of_plot:
        plt.savefig('figures/'+filename_txt+'_multi_temp_maps_'+region+'_proxies_'+str(show_proxies)+'_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

multi_map('NAmerica','PaperFig10',show_proxies='values_gridded_ann')
multi_map('Europe',  'PaperFig11',show_proxies='values_gridded_ann')

"""
for show_proxies in ['circles','values','values_annual','values_gridded','values_gridded_ann']:
    multi_map('NAmerica','extra_Fig6',show_proxies=show_proxies)
    multi_map('Europe',  'extra_FigS5',show_proxies=show_proxies)
    multi_map('India',   'extra_Fig6',show_proxies=show_proxies)
    multi_map('NAfrica', 'extra_Fig6',show_proxies=show_proxies)
"""

