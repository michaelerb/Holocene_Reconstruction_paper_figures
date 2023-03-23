#==============================================================================
# Interpolate all records to decadal resolution, then make a shaded plot of all
# of them from north to south. It also compares plots like this to the zonal-
# mean reconstruction. It makes Figs. 2, 4, 8 and calcuations for Table 1 in
# the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/data_assimilation')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import da_load_proxies

save_instead_of_plot = False
standardize_data     = True

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]


#%% LOAD DATA

# Load the proxies a different way
options = {}
options['data_dir']            = '/projects/pd_lab/data/data_assimilation/'
options['time_resolution']     = 10
options['maximum_resolution']  = 1000
options['verbose_level']       = 0
options['assign_seasonality']  = False
options['reconstruction_type'] = 'relative'
options['proxy_datasets_to_assimilate'] = ['temp12k']
options['age_range_to_reconstruct']     = [0,12000]
options['reference_period']             = [3000,5000]

# Load and process the proxy data
filtered_ts,collection_all = da_load_proxies.load_proxies(options)
proxy_data = da_load_proxies.process_proxies(filtered_ts,collection_all,options)
proxy_values   = np.transpose(proxy_data['values_binned'])
proxy_metadata = proxy_data['metadata']
ages_binned    = proxy_data['age_centers']


#%%
# Load a Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
recon_mean = handle['recon_tas_mean'].values
ages_da    = handle['ages'].values
lat        = handle['lat'].values
lon        = handle['lon'].values
handle.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Get proxy metadata
n_proxies = proxy_values.shape[1]
n_time    = proxy_values.shape[0]
datasetname_all = proxy_metadata[:,0]
tsid_all        = proxy_metadata[:,1]
lats_all        = proxy_metadata[:,2].astype(float)
lons_all        = proxy_metadata[:,3].astype(float)
seasonality_all = proxy_metadata[:,5]
n_proxies_valid = np.sum(np.sum(np.isfinite(proxy_values),axis=0) > 0)

print('Recon lons:',min(lon),max(lon))
print('Proxy lons:',min(lons_all),max(lons_all))
lon[lon > 180] = lon[lon > 180] - 360
lons_all[lons_all > 180] = lons_all[lons_all > 180] - 360

# Sort the lats from north to south.
sort_ind = np.argsort(-lats_all)
lats_sorted           = np.array(lats_all)[sort_ind]
lons_sorted           = np.array(lons_all)[sort_ind]
proxy_values_sorted   = proxy_values[:,sort_ind]
proxy_metadata_sorted = proxy_metadata[sort_ind,:]
datasetname_sorted    = np.array(datasetname_all)[sort_ind]
tsid_sorted           = np.array(tsid_all)[sort_ind]
seasonality_sorted    = np.array(seasonality_all)[sort_ind]
filtered_ts_sorted    = np.array(filtered_ts)[sort_ind]

# Create a new array, where each record is placed accorcing to its latitude.
# Where multiple records exist for the same latitude, average them together
lat_bound_resolution = 0.5
lat_bounds = np.arange(90,-90.1,-lat_bound_resolution)
lat_centers = (lat_bounds[:-1]+lat_bounds[1:])/2

# Find indices closest to latitudes
lats_to_label = np.array([80,70,60,50,40,30,20,10,0,-10,-30,-50,-70])
inds_to_label = np.zeros(len(lats_to_label)); inds_to_label[:] = np.nan
for i,lat_selected in enumerate(lats_to_label):
    inds_to_label[i] = np.argmin(np.abs(lats_sorted - lat_selected))

# Separate proxies into all, all_unique, annual_only, summer_only, and winter_only
ind_season = {}
ind_season['all']    = n_proxies*[True]
ind_season['annual'] = seasonality_sorted == 'annual'
ind_season['summer'] = (seasonality_sorted == 'summer+') | (seasonality_sorted == 'summerOnly')
ind_season['winter'] = (seasonality_sorted == 'winter+') | (seasonality_sorted == 'winterOnly')
print(sum(ind_season['annual']),sum(ind_season['summer']),sum(ind_season['winter']))

# Find the valid proxies
ind_valid = np.all(np.isfinite(proxy_values),axis=0)
print('Records with valid data:',sum(ind_valid))
print(' === Percentages of different seasonalities ===')
for season in ['annual','summer','winter']:
    print(season,':',(sum(ind_season[season] & ind_valid)/sum(ind_valid))*100)

lat_da_edges = (lat[1:] + lat[:-1])/2
lat_da_edges = np.insert(lat_da_edges,0,(lat_da_edges[0] - (lat[1] - lat[0])))
lat_da_edges = np.append(lat_da_edges,(lat_da_edges[-1] + (lat[-1] - lat[-2])))


#%%
# Calculate the percentage of records which have a warming trend over the interval
season,age_range = 'annual',[12000,6000]
def percent_warming(season,age_range):
    #
    filtered_ts_selected = filtered_ts_sorted[ind_season[season]]
    proxies_total = len(filtered_ts_selected)
    proxies_warming = 0
    proxies_cooling = 0
    proxies_flat    = 0
    proxies_insuf   = 0
    i=0
    for i in range(proxies_total):
        #
        # Get proxy data
        proxy_ages = np.array(filtered_ts_selected[i]['age']).astype(float)
        proxy_values = np.array(filtered_ts_selected[i]['paleoData_values']).astype(float)
        #
        ind_selected = (proxy_ages <= age_range[0]) & (proxy_ages > age_range[1]) & np.isfinite(proxy_values)
        data_in_range = proxy_values[ind_selected]
        ages_in_range = proxy_ages[ind_selected]
        ages_in_range_ka = -1*ages_in_range/1000
        #
        n_necessary = 5
        if len(ages_in_range) < n_necessary: proxies_insuf+=1; continue
        slope,intercept,rvalue,pvalue,_ = stats.linregress(ages_in_range_ka,data_in_range)
        #
        if pvalue > 0.05: proxies_flat += 1;    title_txt = 'flat'
        elif slope > 0:   proxies_warming += 1; title_txt = 'warmimg'
        elif slope < 0:   proxies_cooling += 1; title_txt = 'cooling'
        #
        """
        plt.plot(ages_in_range_ka,data_in_range,'k-')
        plt.plot(ages_in_range_ka,(slope*ages_in_range_ka)+intercept,'b-')
        plt.title(title_txt+', slope='+str(slope)+', p-value'+str(pvalue))
        plt.show()
        input('Press enter to continue')
        """
    #
    proxies_sum       = proxies_warming+proxies_cooling+proxies_flat+proxies_insuf
    proxies_sum_valid = proxies_warming+proxies_cooling+proxies_flat
    if proxies_total != proxies_sum: print('Total and sum differ:',proxies_total,proxies_sum)
    #
    print(' -- Proxy trends for '+season+' season during the period '+str(age_range)+' --')
    print('Total:             ',proxies_total)
    print('Total used:        ',proxies_sum_valid)
    print('Warming:           ',proxies_warming,' | ',str('{:.1f}'.format((proxies_warming/proxies_sum_valid)*100)),'%')
    print('Cooling:           ',proxies_cooling,' | ',str('{:.1f}'.format((proxies_cooling/proxies_sum_valid)*100)),'%')
    print('Flat:              ',proxies_flat,' | ',str('{:.1f}'.format((proxies_flat/proxies_sum_valid)*100)),'%')
    print('Less than '+str(n_necessary)+' points:',proxies_insuf)

print(' === TABLE 1 VALUES ===')

for season in ['all','annual','summer','winter']:
    percent_warming(season,[12000,6000])

for season in ['all','annual','summer','winter']:
    percent_warming(season,[6000,0])


#%%
# Find the warmest and coldest periods averaged over a number of records
data_selected,ages_selected,lats_selected,standardize = proxy_values_sorted,ages_da,lats_sorted,False
data_selected,ages_selected,lats_selected,standardize = proxy_values_sorted[:,ind_season[season]],ages_da,lats_sorted[ind_season[season]],standardize_data
#data_selected,ages_selected,lats_selected,standardize = data_12ka_by_lat,ages_da,lat_centers,standardize_data
def timing_extremes(data_selected,ages_selected,lats_selected,standardize):
    #
    # If selected, standardize all data
    if standardize: data_selected = (data_selected - np.nanmean(data_selected,axis=0)) / np.nanstd(data_selected,axis=0)
    #
    # Set the latitude bounds to average over
    lat_bounds = np.arange(90,-91,-30)
    n_regions = len(lat_bounds)-1
    minimum_fraction = 0.25
    #
    # Calculate timing of max and min temperature for all proxies.
    n_ts = data_selected.shape[1]
    mean_of_data = np.nanmean(data_selected,axis=1)
    nvalid_ts = np.sum(np.isfinite(data_selected),axis=1)
    mean_of_data[nvalid_ts < (n_ts*minimum_fraction)] = np.nan
    if sum(np.isfinite(mean_of_data)) > 0:
        age_warmest = ages_selected[np.nanargmax(mean_of_data)]
        age_coldest = ages_selected[np.nanargmin(mean_of_data)]
    else:
        age_warmest = np.nan
        age_coldest = np.nan
    #
    # Set up arrays
    age_warmest_region = np.zeros((n_ts)); age_warmest_region[:] = np.nan
    age_coldest_region = np.zeros((n_ts)); age_coldest_region[:] = np.nan
    for i in range(n_regions):
        ind_selected   = np.where((lats_selected <= lat_bounds[i]) & (lats_selected > lat_bounds[i+1]))[0]
        mean_of_data = np.nanmean(data_selected[:,ind_selected],axis=1)
        nvalid_ts = np.sum(np.isfinite(data_selected[:,ind_selected]),axis=1)
        mean_of_data[nvalid_ts < (len(ind_selected)*minimum_fraction)] = np.nan
        #
        if sum(np.isfinite(mean_of_data)) > 0:
            age_warmest_region[ind_selected] = ages_selected[np.nanargmax(mean_of_data)]
            age_coldest_region[ind_selected] = ages_selected[np.nanargmin(mean_of_data)]
    #
    return age_warmest_region,age_coldest_region,age_warmest,age_coldest


#%% FIGURES
plt.style.use('ggplot')

max_value = 2

# Make a plot of all records
plt.figure(figsize=(28,24))
ax1 = plt.subplot2grid((1,1),(0,0))

# Define the edges for the time, lat, and proxy number
proxy_counter       = np.arange(n_proxies)
proxy_counter_edges = np.arange(n_proxies+1)+0.5
ages_da_edges = (ages_da[1:] + ages_da[:-1])/2
ages_da_edges = np.insert(ages_da_edges,0,(ages_da_edges[0] - (ages_da[1] - ages_da[0])))
ages_da_edges = np.append(ages_da_edges,(ages_da_edges[-1] + (ages_da[-1] - ages_da[-2])))

age_warmest_proxies,age_coldest_proxies,age_warmest,age_coldest = timing_extremes(proxy_values_sorted,ages_da,lats_sorted,standardize=standardize_data)
figure = ax1.pcolormesh(ages_da_edges,proxy_counter_edges,np.transpose(proxy_values_sorted),cmap='bwr',vmin=-max_value,vmax=max_value)
ax1.scatter(age_warmest_proxies,proxy_counter,20,c='k')
colorbar = plt.colorbar(figure,orientation='horizontal',aspect=40,pad=0.07)
colorbar.set_label('$\Delta$Temperature ($^\circ$C)',fontsize=32)
colorbar.ax.tick_params(labelsize=28)
ax1.set_xlim(12000,0)
ax1.set_ylim(proxy_counter_edges[0],proxy_counter_edges[-1])
ax1.set_xlabel('Age (yr BP)',fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Make two y axes
ax1.invert_yaxis()
ax1.set_ylabel('Proxy number',fontsize=32)
ax1.yaxis.set_major_locator(MultipleLocator(100))
ax1.yaxis.set_minor_locator(MultipleLocator(10))

ymin,ymax = ax1.get_ylim()
ax2 = ax1.twinx()
ax2.set_ylim(ymin,ymax)
ax2.set_yticks(inds_to_label+1)
ax2.set_yticklabels(lats_to_label,fontsize=28)
ax2.set_ylabel('Latitude ($^\circ$)',fontsize=32)
ax2.grid(False)

plt.tight_layout()
plt.suptitle('Decadal temperature anomalies ($^\circ$C)\nfor all proxies arranged from north to south, $N_{validproxies}$ = '+str(n_proxies_valid),fontsize=50)
#plt.suptitle('Decadal temperature anomalies ($^\circ$C)\nfor all proxies arranged from north to south, $N_{validproxies}$ = '+str(n_proxies_valid)+'\nwarmest age: '+str(age_warmest)+', coldest age: '+str(age_coldest),fontsize=50)
plt.subplots_adjust(top=.89)

if save_instead_of_plot:
    plt.savefig('figures/PaperFig2_proxy_decadal_da_overview_all.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


#%%
# Make a plot of all records
#f, ax = plt.subplots(3,1,figsize=(14,20),sharex=True)
fig_seasons = plt.figure(figsize=(28,40))
ax,ax_twin = {},{}

# Define subplots
widths = [1,1,1]
heights= [.4,1.2,.9,.7]
fig_spec = fig_seasons.add_gridspec(ncols=3,nrows=4,width_ratios=widths,height_ratios=heights)
ax['map0'] = fig_seasons.add_subplot(fig_spec[0,0],projection=ccrs.Robinson()); ax['map0'].set_global()
ax['map1'] = fig_seasons.add_subplot(fig_spec[0,1],projection=ccrs.Robinson()); ax['map1'].set_global()
ax['map2'] = fig_seasons.add_subplot(fig_spec[0,2],projection=ccrs.Robinson()); ax['map2'].set_global()

# Make a map
letters = ['a','b','c']
colors = ['k','r','b']
for i,season in enumerate(['annual','summer','winter']):
    ind_valid = np.sum(np.isfinite(proxy_values_sorted),axis=0) > 0
    ind_valid_season = ind_valid & ind_season[season]
    ax['map'+str(i)].scatter(lons_sorted[ind_valid_season],lats_sorted[ind_valid_season],30,c='r',marker='o',edgecolor='k',alpha=1,linewidths=1,transform=ccrs.PlateCarree(),label=season.capitalize())
    ax['map'+str(i)].coastlines()
    ax['map'+str(i)].add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax['map'+str(i)].gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax['map'+str(i)].spines['geo'].set_edgecolor('black')
    ax['map'+str(i)].set_title('('+letters[i]+') '+season.capitalize()+' proxies',loc='center',fontsize=36)

letters = ['d','e','f']
for i,season in enumerate(['annual','summer','winter']):
    #
    # Define the edges for the time, lat, and proxy number
    ax[i] = fig_seasons.add_subplot(fig_spec[(i+1),:])
    n_proxies_season = sum(ind_season[season])
    proxy_counter       = np.arange(n_proxies_season)+1
    proxy_counter_edges = np.arange(n_proxies_season+1)+0.5
    age_warmest_proxies,age_coldest_proxies,age_warmest,age_coldest = timing_extremes(proxy_values_sorted[:,ind_season[season]],ages_da,lats_sorted[ind_season[season]],standardize=standardize_data)
    figure = ax[i].pcolormesh(ages_da_edges,proxy_counter_edges,np.transpose(proxy_values_sorted[:,ind_season[season]]),cmap='bwr',vmin=-max_value,vmax=max_value)
    ax[i].scatter(age_warmest_proxies,proxy_counter,20,c='k')
    if i == 2:
        colorbar = plt.colorbar(figure,orientation='horizontal',aspect=40,pad=0.15)
        colorbar.set_label('$\Delta$Temperature ($^\circ$C)',fontsize=32)
        colorbar.ax.tick_params(labelsize=28)
    ax[i].set_xlim(12000,0)
    ax[i].set_xlabel('Age (yr BP)',fontsize=32)
    ax[i].set_title('('+letters[i]+') '+season.capitalize()+', proxy time series, N$_{proxies}$ = '+str(n_proxies_season),fontsize=36,loc='left')
    #ax[i].set_title('('+letters[i]+') '+season.capitalize()+', proxy time series, N$_{proxies}$ = '+str(n_proxies_season)+', warmest age: '+str(age_warmest)+', coldest age: '+str(age_coldest),fontsize=36,loc='left')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    #
    # Make two y axes
    ax[i].invert_yaxis()
    ax[i].set_ylabel('Proxy number',fontsize=32)
    ax[i].yaxis.set_major_locator(MultipleLocator(100))
    ax[i].yaxis.set_minor_locator(MultipleLocator(10))
    #
    ymin,ymax = ax[i].get_ylim()
    ax_twin[i] = ax[i].twinx()
    ax_twin[i].set_ylim(ymin,ymax)
    ax_twin[i].set_yticks(inds_to_label+1)
    ax_twin[i].set_yticklabels(lats_to_label,fontsize=28)
    ax_twin[i].set_ylabel('Latitude ($^\circ$)',fontsize=32)
    ax_twin[i].grid(False)

plt.tight_layout()
plt.suptitle('Decadal temperature anomalies ($^\circ$C)\nfor all proxies arranged by season and north to south',fontsize=50)
plt.subplots_adjust(top=.92)

if save_instead_of_plot:
    plt.savefig('figures/PaperFig4_proxy_decadal_da_overview_seasonal.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


#%%
# A function to bin records by latitude
#indices = indices['annual']
season,lat_min,lat_max,lon_min,lon_max = 'all',-90,90,-180,180
def bin_by_lat(season,lat_min,lat_max,lon_min,lon_max):
    #
    indices = np.where(ind_season[season] & (lats_sorted >= lat_min) & (lats_sorted <= lat_max) & (lons_sorted >= lon_min) & (lons_sorted <= lon_max))[0]
    #
    data_12ka_by_lat = np.zeros((n_time,len(lat_centers))); data_12ka_by_lat[:] = np.nan
    nproxy_per_lat   = np.zeros((len(lat_centers)))
    for i in range(len(lat_centers)):
        inds_in_lat = np.where((lats_sorted <= lat_bounds[i]) & (lats_sorted > lat_bounds[i+1]))[0]
        inds_selected = np.array(list(set(inds_in_lat) & set(indices)))
        #print(lat_bounds[i],lat_bounds[i+1],len(inds_selected))
        if len(inds_selected) > 0:
            data_12ka_by_lat[:,i] = np.nanmean(proxy_values_sorted[:,inds_selected],axis=1)
            nproxy_per_lat[i] = len(inds_selected)
    #
    n_proxies_valid_binned = np.sum(np.sum(np.isfinite(proxy_values_sorted[:,indices]),axis=0) > 0)
    #
    return data_12ka_by_lat,nproxy_per_lat,n_proxies_valid_binned


#%%
# Make a plot of all records, grouped by latitude
season,lat_min,lat_max,lon_min,lon_max,filename_txt_1,filename_txt_2 = 'annual',-90,90,-180,180,'test','test'
def plot_proxies_and_recon(season,lat_min,lat_max,lon_min,lon_max,filename_txt_1,filename_txt_2):
    #
    # Bin records with certain seasons
    data_12ka_by_lat,nproxy_per_lat,n_proxies_used = bin_by_lat(season,lat_min,lat_max,lon_min,lon_max)
    #
    plt.figure(figsize=(27,18))
    ax1 = plt.subplot2grid((1,2),(0,0))
    figure_proxies = ax1.pcolormesh(ages_da_edges,lat_bounds,np.transpose(data_12ka_by_lat),cmap='bwr',vmin=-max_value,vmax=max_value)
    age_warmest_proxies,_,age_warmest,age_coldest = timing_extremes(data_12ka_by_lat,ages_da,lat_centers,standardize=standardize_data)
    ax1.scatter(age_warmest_proxies,lat_centers,20,c='k')
    ax1.set_xlim(12000,0)
    ax1.set_ylim(lat_min,lat_max)
    ax1.set_xlabel('Age (yr BP)',fontsize=27)
    ax1.set_ylabel('Latitude ($^\circ$)',fontsize=27)
    ax1.set_title('(a) '+season.capitalize()+' proxies, binned by latitude, $N_{validproxies}$ = '+str(n_proxies_used),loc='left',fontsize=30)
    #ax1.set_title('(a) '+season.capitalize()+' proxies, binned by latitude, $N_{validproxies}$ = '+str(n_proxies_used)+'\nwarmest age: '+str(age_warmest)+', coldest age: '+str(age_coldest),loc='left',fontsize=30)
    colorbar1 = plt.colorbar(figure_proxies,orientation='horizontal',label='Temperature',ax=ax1,fraction=0.08,pad=0.07)
    colorbar1.set_label('$\Delta$Temperature ($^\circ$C)',fontsize=27)
    colorbar1.ax.tick_params(labelsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    #
    # Plot reconstruction averaged over the selected longitudes
    ax2 = plt.subplot2grid((1,2),(0,1))
    ind_recon = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    recon_to_plot = np.mean(recon_mean[:,:,ind_recon],axis=2)
    figure_da = ax2.pcolormesh(ages_da_edges,lat_da_edges,np.transpose(recon_to_plot),cmap='bwr',vmin=-max_value,vmax=max_value)
    age_warmest_da,_,age_warmest,age_coldest = timing_extremes(recon_to_plot,ages_da,lat,standardize=standardize_data)
    ax2.scatter(age_warmest_da,lat,20,c='k')
    ax2.set_xlim(12000,0)
    ax2.set_ylim(lat_min,lat_max)
    ax2.set_xlabel('Age (yr BP)',fontsize=27)
    ax2.set_ylabel('Latitude ($^\circ$)',fontsize=27)
    ax2.set_title('(b) Zonal-mean Holocene reconstruction',loc='left',fontsize=30)
    #ax2.set_title('(b) Zonal-mean Holocene reconstruction\nwarmest age: '+str(age_warmest)+', coldest age: '+str(age_coldest),loc='left',fontsize=30)
    colorbar2 = plt.colorbar(figure_da,orientation='horizontal',label='Temperature',ax=ax2,fraction=0.08,pad=0.07)
    colorbar2.set_label('$\Delta$Temperature ($^\circ$C)',fontsize=27)
    colorbar2.ax.tick_params(labelsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    #
    plt.tight_layout()
    #plt.suptitle('Decadal temperature anomalies ($\Delta^\circ$C), lats='+str(lat_min)+'-'+str(lat_max)+', lons='+str(lon_min)+'-'+str(lon_max),fontsize=37)
    #plt.subplots_adjust(top=.9)
    #
    if save_instead_of_plot:
        plt.savefig('figures/'+filename_txt_1+'_proxy_vs_recon_'+season+'_'+filename_txt_2+'_'+exp_txt+'.png',dpi=300,format='png')
        plt.close()
    else:
        plt.show()


plot_proxies_and_recon('annual',-90,90,-180,180,'PaperFig8','1_global')
plot_proxies_and_recon('all',   -90,90,-180,180,'extra_Fig8','1_global')

