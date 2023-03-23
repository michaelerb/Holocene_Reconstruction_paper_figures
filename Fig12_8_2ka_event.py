#==============================================================================
# Make a figure to explore the 8.2 ka event in the reconstruction. This is
# Fig. 12 in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/data_assimilation')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import xarray as xr
import copy
import da_utils
import da_load_proxies
from matplotlib import colors


save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]

ages_anom = [8210,8100]
ages_ref1 = [8300,8210]
ages_ref2 = [8100,8000]


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
gmt_recon_all   = handle['recon_tas_global_mean'].values
recon_mean      = handle['recon_tas_mean'].values
recon_ens       = handle['recon_tas_ens'].values
ages_da         = handle['ages'].values
lat             = handle['lat'].values
lon             = handle['lon'].values
proxy_values    = handle['proxy_values'].values
proxy_metadata  = handle['proxy_metadata'].values
proxy_assim_all = handle['proxies_assimilated'].values
options_da      = handle['options'].values
handle.close()


#%%
# Get the options
options = {}
keys_to_ints_list = ['age_range_to_reconstruct','reference_period','age_range_model']
for i in range(len(options_da)):
    text,value = options_da[i].split(':')
    if '[' in value: value = value.replace('[','').replace(']','').replace("'",'').replace(' ','').split(',')
    if text in keys_to_ints_list: value = [int(i) for i in value]
    if (isinstance(value,str)) and (value.isdecimal()): value = int(value)
    options[text] = value

# Load the original proxy data
filtered_ts,_ = da_load_proxies.load_proxies(options)


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Get values of assimilated proxies
ind_proxies_assimilated = np.where(np.mean(proxy_assim_all,axis=0) > 0)[0]
ind_proxies_leftout     = np.where(np.mean(proxy_assim_all,axis=0) == 0)[0]
proxy_values_assim = copy.deepcopy(proxy_values)
proxy_values_assim[:,ind_proxies_leftout] = np.nan

# Get proxy lats and lons
proxy_lats = proxy_metadata[:,2].astype(float)
proxy_lons = proxy_metadata[:,3].astype(float)
proxy_season = proxy_metadata[:,5]

# Make lon run from -180 to 180
ind_east = np.where(lon < 180)[0]
ind_west = np.where(lon >= 180)[0]
recon_mean = np.concatenate((recon_mean[:,:,ind_west], recon_mean[:,:,ind_east]), axis=2)
recon_ens  = np.concatenate((recon_ens[:,:,:,ind_west],recon_ens[:,:,:,ind_east]),axis=3)
lon = np.concatenate((lon[ind_west]-360,lon[ind_east]),axis=0)


#%%
# Find proxies in the greenland region
ind_greenland = []
n_proxies = proxy_metadata.shape[0]
for i in range(n_proxies):
    if proxy_metadata[i,0] in ['Alley.GISP2.2000','GISP2.Kobashi.2017','NorthGRIP.Gkinis.2014']: ind_greenland.append(i)


#%%
# Select three sample ensemble members
n_ens = gmt_recon_all.shape[1]
np.random.seed(seed=0)
ind_random = np.random.choice(n_ens,3,replace=False)

# Compute the anomalies
indices_anom = np.where((ages_da <= ages_anom[0]) & (ages_da >= ages_anom[1]))[0]
indices_ref1 = np.where((ages_da <= ages_ref1[0]) & (ages_da >= ages_ref1[1]))[0]
indices_ref2 = np.where((ages_da <= ages_ref2[0]) & (ages_da >= ages_ref2[1]))[0]
indices_ref = np.concatenate((indices_ref1,indices_ref2))

gmt_8_2ka          = np.nanmean(gmt_recon_all[indices_anom,:],axis=0)      - np.nanmean(gmt_recon_all[indices_ref,:],axis=0)
recon_ens_8_2ka    = np.nanmean(recon_ens[indices_anom,:,:,:],axis=0)      - np.nanmean(recon_ens[indices_ref,:,:,:],axis=0)
recon_mean_8_2ka   = np.nanmean(recon_mean[indices_anom,:,:],axis=0)       - np.nanmean(recon_mean[indices_ref,:,:],axis=0)
proxy_values_8_2ka = np.nanmean(proxy_values_assim[indices_anom,:],axis=0) - np.nanmean(proxy_values_assim[indices_ref,:],axis=0)

"""
def valid_inds(var):
    value1 = np.nanmean(var[indices_anom,:],axis=0)
    value2 = np.nanmean(var[indices_ref1,:],axis=0)
    value3 = np.nanmean(var[indices_ref2,:],axis=0)
    valid = np.isfinite(value1) & np.isfinite(value2) & np.isfinite(value3) & (value1 != value2) & (value1 != value3)
    return valid

# Only plot proxies which have values in all three periods
ind_valid = valid_inds(proxy_values_assim)
"""

# Find proxies with at least one original proxy value in each interval.
ind_valid = []
for i in ind_proxies_assimilated:
    #
    # Get proxy data
    proxy_ages   = np.array(filtered_ts[i]['age']).astype(float)
    proxy_values = np.array(filtered_ts[i]['paleoData_values']).astype(float)
    #
    # If any NaNs exist in the ages or values, remove those ages
    ind_values_valid = np.isfinite(proxy_ages) & np.isfinite(proxy_values)
    proxy_ages = proxy_ages[ind_values_valid]
    #
    # Check for values in the three intervals
    n_values_anom = len(np.where((proxy_ages <= ages_anom[0]) & (proxy_ages >= ages_anom[1]))[0])
    n_values_ref1 = len(np.where((proxy_ages <= ages_ref1[0]) & (proxy_ages >= ages_ref1[1]))[0])
    n_values_ref2 = len(np.where((proxy_ages <= ages_ref2[0]) & (proxy_ages >= ages_ref2[1]))[0])
    if (n_values_anom > 0) & (n_values_ref1 > 0) & (n_values_ref2 > 0):
        ind_valid.append(i)

ind_valid = np.array(ind_valid)


#%%
# Print the percentage of records with different values in the three different intervals
n_proxies_assim = sum(np.isfinite(np.nanmean(proxy_values_assim,axis=0)))
n_proxies_valid = len(ind_valid)
print('Assimilated proxies = '+str(n_proxies_assim))
print('Different values in each period = '+str(n_proxies_valid))
print(' --- Percentage of assimiated records with valid data = '+str((n_proxies_valid/n_proxies_assim)*100))


#%%
# Compute means over different regions
greenland_region = [60,85,-75,-15]
europe_region    = [40,70,-10,35]
greenland_8_2ka = da_utils.spatial_mean(recon_ens_8_2ka,lat,lon,greenland_region[0],greenland_region[1],greenland_region[2],greenland_region[3],1,2)
europe_8_2ka    = da_utils.spatial_mean(recon_ens_8_2ka,lat,lon,europe_region[0],   europe_region[1],   europe_region[2],   europe_region[3],   1,2)

print('Ranges')
print('GMT:      ',min(gmt_8_2ka),max(gmt_8_2ka))
print('Greenland:',min(greenland_8_2ka),max(greenland_8_2ka))
print('Europe:   ',min(europe_8_2ka),max(europe_8_2ka))


#%% FIGURES
plt.style.use('ggplot')

# Make a time series and map for the paper
plt.figure(figsize=(16,20))
ax1 = plt.subplot2grid((4,1),(0,0))
ax1.fill_between(ages_da,np.nanmin(gmt_recon_all,axis=1),np.nanmax(gmt_recon_all,axis=1),color='k',alpha=0.25)
ax1.plot(ages_da,np.nanmean(gmt_recon_all,axis=1),color='k',linewidth=3,label='Mean')
colors_to_plot = ['k','b','r']
for i,ind in enumerate(ind_random):
    ax1.plot(ages_da,gmt_recon_all[:,ind],color=colors_to_plot[i],linewidth=1,label='Ens. '+str(ind))
#for i,ind in enumerate(ind_greenland):
#    ax1.plot(ages_da,proxy_values[:,ind],color=colors_to_plot[i],linewidth=1,label=proxy_metadata[i,0])

ax1.legend(fontsize=16,ncol=2,loc='lower right')
ax1.axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
ax1.axvspan(ages_anom[0],ages_anom[1],alpha=0.25,color='tab:blue')
ax1.axvspan(ages_ref1[0],ages_ref1[1],alpha=0.25,color='tab:red')
ax1.axvspan(ages_ref2[0],ages_ref2[1],alpha=0.25,color='tab:red')
ax1.set_xlim(8700,7600)
ax1.set_ylim(-0.55,0.15)
#ax1.set_ylim(-4,7)
ax1.set_xlabel('Age (yr BP)',fontsize=20)
ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=20)
ax1.tick_params(axis='both',which='major',labelsize=20)
ax1.set_title('(a) Global-mean temperature anomaly ($^\circ$C)',fontsize=24,loc='left')

# Make a map of changes
region = 'global'
if region == 'global':
    ax2 = plt.subplot2grid((4,1),(1,0),rowspan=2,projection=ccrs.Robinson(central_longitude=0)); ax2.set_global()
    marker_size = 50
elif region == 'NA_and_Greenland':
    ax2 = plt.subplot2grid((4,1),(1,0),rowspan=2,projection=ccrs.PlateCarree()); ax2.set_extent([-140,0,30,90],crs=ccrs.PlateCarree())
    marker_size = 100

# Create the same colorbar for both contourf and scatter
anomaly_value = 1
range_selected = np.linspace(-anomaly_value,anomaly_value,21)
colors_discrete = colors.BoundaryNorm(range_selected,plt.cm.bwr.N)

recon_mean_8_2ka_cyclic,lon_cyclic = cutil.add_cyclic_point(recon_mean_8_2ka,coord=lon)
map1 = ax2.contourf(lon_cyclic,lat,recon_mean_8_2ka_cyclic,range_selected,extend='both',norm=colors_discrete,cmap='bwr',transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
ax2.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
ax2.spines['geo'].set_edgecolor('black')
ax2.set_title('(b) Map of temperature anomalies ($^\circ$C)',fontsize=24,loc='left')

for i in ind_valid:
    if   (proxy_season[i] == 'annual'):                                      marker_symbol = 'o'
    elif (proxy_season[i] == 'summerOnly') | (proxy_season[i] == 'summer+'): marker_symbol = '^'
    elif (proxy_season[i] == 'winterOnly') | (proxy_season[i] == 'winter+'): marker_symbol = 'v'
    if np.isnan(proxy_values_8_2ka[i]): continue
    ax2.scatter([proxy_lons[i]],[proxy_lats[i]],marker_size,c=[proxy_values_8_2ka[i]],marker=marker_symbol,edgecolor='k',alpha=1,cmap='bwr',norm=colors_discrete,linewidths=1,transform=ccrs.PlateCarree())

colorbar = plt.colorbar(map1,orientation='horizontal',ax=ax2,fraction=0.08,aspect=40,pad=0.02)
colorbar.set_label('$\Delta$Temperature ($^\circ$C)',fontsize=20)
colorbar.ax.tick_params(labelsize=20)
colorbar.ax.set_facecolor('none')

# Draw boxes around the regions selected for analysis
ax2.add_patch(mpatches.Rectangle(xy=[greenland_region[2],greenland_region[0]],width=(greenland_region[3]-greenland_region[2]),height=(greenland_region[1]-greenland_region[0]),edgecolor='k',fill=False,linewidth=1,transform=ccrs.PlateCarree()))
ax2.add_patch(mpatches.Rectangle(xy=[europe_region[2],   europe_region[0]],   width=(europe_region[3]-europe_region[2]),      height=(europe_region[1]-europe_region[0]),      edgecolor='k',fill=False,linewidth=1,transform=ccrs.PlateCarree()))

# Histogram of changes
ax3 = plt.subplot2grid((4,1),(3,0))
bins = np.arange(-1,1.01,.025)
ax3.hist(gmt_8_2ka,      bins=bins,color='k',         alpha=0.5,zorder=2,weights=np.ones(len(gmt_8_2ka))       / len(gmt_8_2ka),      label='Global mean (n='+str(len(gmt_8_2ka))+'),      mean = '+str('{:.2f}'.format(np.nanmean(gmt_8_2ka)))+'$^\circ$C')
ax3.hist(greenland_8_2ka,bins=bins,color='tab:green', alpha=0.5,zorder=2,weights=np.ones(len(greenland_8_2ka)) / len(greenland_8_2ka),label='Greenland region (n='+str(len(greenland_8_2ka))+'), mean = '+str('{:.2f}'.format(np.nanmean(greenland_8_2ka)))+'$^\circ$C')
ax3.hist(europe_8_2ka,   bins=bins,color='tab:purple',alpha=0.5,zorder=2,weights=np.ones(len(europe_8_2ka))    / len(europe_8_2ka),   label='Europe region (n='+str(len(europe_8_2ka))+'),      mean = '+str('{:.2f}'.format(np.nanmean(europe_8_2ka)))+'$^\circ$C')

ax3.axvline(x=np.nanmean(gmt_8_2ka),      color='k',         alpha=1,linestyle=':',linewidth=2)
ax3.axvline(x=np.nanmean(greenland_8_2ka),color='tab:green', alpha=1,linestyle=':',linewidth=2)
ax3.axvline(x=np.nanmean(europe_8_2ka),   color='tab:purple',alpha=1,linestyle=':',linewidth=2)
ax3.axvline(x=0,color='gray',alpha=1,linestyle='--',linewidth=3)

ax3.legend(fontsize=16,loc='upper right')
ax3.set_xlim(-1,1)
ax3.set_ylabel('Percentage (%)',fontsize=20)
ax3.set_xlabel('$\Delta$Temperature ($^\circ$C)',fontsize=20)
ax3.tick_params(axis='both',which='major',labelsize=20)
ax3.set_title('(c) Regional-mean temperature anomalies ($^\circ$C)',fontsize=24,loc='left')

plt.tight_layout()
plt.suptitle('Temperature anomaly near 8.2 ka',fontsize=28)
plt.subplots_adjust(top=.93)

if save_instead_of_plot:
    plt.savefig('figures/PaperFig12_8_2ka_event_'+region+'_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()

