#==============================================================================
# Analyze output from the pseudoproxy reconstruction. This is Fig. A1 in the
# paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/data_assimilation')
import da_utils
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import xarray as xr
import netCDF4

save_instead_of_plot = False
make_output          = False
plt.style.use('ggplot')

# Pick the reconstruction to analyze
recon_filename = 'holocene_recon_2022-04-04_16:55:44.482606_pseudo_default.nc'  # The default pseudoproxy experiment: 1-model (hadcm3) reconstructing trace
#recon_filename = sys.argv[1]

# Choose which value to plot
map_value_name = 'Correlation'
#map_value_name = 'CE'


#%% LOAD DATA

#vars_all = ['tas','precip']
vars_all = ['tas']

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
recon_global,prior_global,recon_spatial = {},{},{}
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
for var_name in vars_all:
    recon_global[var_name]  = handle['recon_'+var_name+'_global_mean'].values
    prior_global[var_name]  = handle['prior_'+var_name+'_global_mean'].values
    recon_spatial[var_name] = handle['recon_'+var_name+'_mean'].values

recon_mean_proxies = handle['proxyrecon_mean'].values
proxy_values   = handle['proxy_values'].values
proxy_metadata = handle['proxy_metadata'].values
ages_da        = handle['ages'].values
lat            = handle['lat'].values
lon            = handle['lon'].values
options_da     = handle['options'].values
handle.close()

# Get the options
options = {}
keys_to_ints_list = ['age_range_to_reconstruct','reference_period','age_range_model']
for i in range(len(options_da)):
    text,value = options_da[i].split(':')
    if '[' in value: value = value.replace('[','').replace(']','').replace("'",'').replace(' ','').split(',')
    if text in keys_to_ints_list: value = [int(i) for i in value]
    if (isinstance(value,str)) and (value.isdecimal()): value = int(value)
    options[text] = value


#%%
# Get information about the pseudoproxy file
proxies_to_use = options['proxy_datasets_to_assimilate'][0].split('_')[1]
model_to_use   = options['proxy_datasets_to_assimilate'][0].split('_')[3]
noise_to_use   = options['proxy_datasets_to_assimilate'][0].split('_')[5]
#time_res = options['time_resolution']
#age_range_model = str(options['age_range_model'][1])+'-'+str(options['age_range_model'][0])

#Note: Regrid the pseudoproxies so that I can test DA experiments using other time resolutions (e.g. 200yr)

# Load the model data
model_dir = '/projects/pd_lab/data/data_assimilation/models/processed_model_data/'
model_filename = model_to_use+'_regrid.21999-0BP.tas.timeres_10.nc'
handle_model = xr.open_dataset(model_dir+model_filename,decode_times=False)
lat_model        = handle_model['lat'].values
lon_model        = handle_model['lon'].values
age_model        = handle_model['age'].values
time_ndays_model = handle_model['days_per_month_all'].values
handle_model.close()

model_data = {}
for var_name in vars_all:
    model_filename = model_to_use+'_regrid.21999-0BP.'+var_name+'.timeres_10.nc'
    handle_model = xr.open_dataset(model_dir+model_filename,decode_times=False)
    model_data[var_name] = handle_model[var_name].values
    handle_model.close()
    model_data[var_name] = np.flip(model_data[var_name],axis=0)

# Flip the data so that it runs from recent to old
age_model        = np.flip(age_model,       axis=0)
time_ndays_model = np.flip(time_ndays_model,axis=0)


#%% CALCULATIONS

# Set some text
title_txt     = {'tas':'temperature','precip':'precipitation'}
unit_txt      = {'tas':'$^\circ$C',  'precip':'$mm day^{-1}$'}
yaxis_txt     = {'tas':'$\Delta$T',  'precip':'$\Delta$P'}
anomaly_value = {'tas':2,            'precip':1}
cmap_selected = {'tas':'bwr',        'precip':'BrBG'}

if   model_to_use == 'hadcm3': model_to_use_txt = 'HadCM3'
elif model_to_use == 'trace':  model_to_use_txt = 'TraCE-21k'
elif model_to_use == 'famous': model_to_use_txt = 'FAMOUS'

# Get some dimensions
n_proxies = proxy_metadata.shape[0]
n_lat     = len(lat)
n_lon     = len(lon)

proxy_names = proxy_metadata[:,0]
proxy_tsids = proxy_metadata[:,1]
proxy_lats  = proxy_metadata[:,2].astype(float)
proxy_lons  = proxy_metadata[:,3].astype(float)
if n_proxies != len(proxy_lats): print('Different numbers of proxies between original file and metadata:',n_proxies,len(proxy_lats))


#%%
# Remove the mean of the reference period
indices_refperiod_recon = np.where((ages_da   >= options['reference_period'][0]) & (ages_da   < options['reference_period'][1]))[0]
indices_refperiod_model = np.where((age_model >= options['reference_period'][0]) & (age_model < options['reference_period'][1]))[0]

gridded_metric,metric_global = {},{}
for var_name in vars_all:
    #recon_spatial[var_name] = recon_spatial[var_name] - np.nanmean(recon_spatial[var_name][indices_refperiod_recon,:,:],axis=0)[None,:,:]
    #recon_global[var_name]  = recon_global[var_name]  - np.nanmean(np.nanmean(recon_global[var_name][indices_refperiod_recon,:],axis=0),axis=0)
    #prior_global[var_name]  = prior_global[var_name]  - np.nanmean(np.nanmean(prior_global[var_name][indices_refperiod_recon,:],axis=0),axis=0)
    model_data[var_name]    = model_data[var_name]    - np.nanmean(model_data[var_name][indices_refperiod_model,:,:],axis=0)[None,:,:]
    #
    # Compute annual means of the model data
    time_ndays_model_latlon = np.repeat(np.repeat(time_ndays_model[:,:,None,None],n_lat,axis=2),n_lon,axis=3)
    model_data[var_name+'_annual'] = np.average(model_data[var_name],axis=1,weights=time_ndays_model_latlon)
    #
    # Compute a global-mean of the model data
    model_data[var_name+'_gmt'] = da_utils.global_mean(model_data[var_name+'_annual'],lat_model,1,2)
    #
    # Get the same years in the model and reconstruction
    indices_selected = np.where((age_model >= options['age_range_to_reconstruct'][0]) & (age_model < options['age_range_to_reconstruct'][1]))[0]
    model_data['age_selected']           = age_model[indices_selected]
    model_data[var_name+'_selected']     = model_data[var_name+'_annual'][indices_selected,:,:]
    model_data[var_name+'_gmt_selected'] = model_data[var_name+'_gmt'][indices_selected]
    #
    # Compute correlation and other metrics between the reconstruction and the model data at every gridpoint
    gridded_metric[var_name] = {}
    gridded_metric[var_name]['Correlation'] = np.zeros((n_lat,n_lon)); gridded_metric[var_name]['Correlation'][:] = np.nan
    gridded_metric[var_name]['Mean_error']  = np.zeros((n_lat,n_lon)); gridded_metric[var_name]['Mean_error'][:]  = np.nan
    gridded_metric[var_name]['RMSE']        = np.zeros((n_lat,n_lon)); gridded_metric[var_name]['RMSE'][:]        = np.nan
    gridded_metric[var_name]['CE']          = np.zeros((n_lat,n_lon)); gridded_metric[var_name]['CE'][:]          = np.nan
    for j in range(n_lat):
        for i in range(n_lon):
            gridded_metric[var_name]['Correlation'][j,i] = np.corrcoef(recon_spatial[var_name][:,j,i],model_data[var_name+'_selected'][:,j,i])[0,1]
            gridded_metric[var_name]['Mean_error'][j,i]  = np.nanmean(np.abs(recon_spatial[var_name][:,j,i]-model_data[var_name+'_selected'][:,j,i]))
            gridded_metric[var_name]['RMSE'][j,i]        = np.sqrt(np.nanmean(np.square(recon_spatial[var_name][:,j,i]-model_data[var_name+'_selected'][:,j,i])))
            gridded_metric[var_name]['CE'][j,i]          = 1 - ( np.sum(np.power(recon_spatial[var_name][:,j,i]-model_data[var_name+'_selected'][:,j,i],2),axis=0) / np.sum(np.power(model_data[var_name+'_selected'][:,j,i]-np.mean(model_data[var_name+'_selected'][:,j,i],axis=0),2),axis=0) )
    #
    # Count data coverage
    nproxies = proxy_values.shape[0]
    data_counts = np.sum(np.isfinite(proxy_values),axis=0)
    #
    # Compute global-means of the metric
    metric_global[var_name] = {}
    for value_name in ['Correlation','Mean_error','RMSE','CE']:
        metric_global[var_name][value_name] = da_utils.global_mean(gridded_metric[var_name][value_name],lat,0,1)


#%% FIGURES
var_name = 'tas'
for var_name in vars_all:
    #
    # Plot the main composite
    plt.figure(figsize=(16,15))
    ax1 = plt.subplot2grid((3,1),(0,0))
    correlation_gmt = np.corrcoef(np.mean(recon_global[var_name],axis=1),model_data[var_name+'_gmt_selected'])[0,1]
    ax1.fill_between(ages_da,np.percentile(prior_global[var_name],2.5,axis=1),np.percentile(prior_global[var_name],97.5,axis=1),color='gray',alpha=0.25)
    ax1.plot(ages_da,np.mean(prior_global[var_name],axis=1),'gray',label='Prior, mean and 95% range')
    ax1.fill_between(ages_da,np.percentile(recon_global[var_name],2.5,axis=1),np.percentile(recon_global[var_name],97.5,axis=1),color='tab:blue',alpha=0.5)
    ax1.plot(ages_da,np.mean(recon_global[var_name],axis=1),'tab:blue',label='Reconstruction, mean and 95% range')
    ax1.plot(age_model,model_data[var_name+'_gmt'],'k',label='Model target ('+model_to_use_txt+')')
    ax1.legend(loc='lower right',fontsize=16)
    ax1.set_ylim(-3.1,0.6)
    ax1.set_xlim(ages_da[-1],ages_da[0])
    ax1.set_ylabel(yaxis_txt[var_name]+' ('+unit_txt[var_name]+')',fontsize=16)
    ax1.set_xlabel('Age (B.P.)',fontsize=16)
    ax1.set_title('(a) Global-mean '+title_txt[var_name]+' ('+unit_txt[var_name]+', R='+str('{:.2f}'.format(correlation_gmt))+')',fontsize=20,loc='left')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #
    # Make a map of values at every point
    ax2 = plt.subplot2grid((3,1),(1,0),rowspan=2,projection=ccrs.Robinson(central_longitude=0))
    if   map_value_name == 'Correlation': bound_min = -1; bound_max = 1; cmap = 'Spectral_r'; extend = 'neither'
    elif map_value_name == 'CE':          bound_min = -1; bound_max = 1; cmap = 'Spectral_r'; extend = 'both'
    elif map_value_name == 'RMSE':        bound_min = 0;  bound_max = 2; cmap = 'viridis';    extend = 'both'
    gridded_metric_cyclic,lon_cyclic = cutil.add_cyclic_point(gridded_metric[var_name][map_value_name],coord=lon)
    map1 = ax2.contourf(lon_cyclic,lat,gridded_metric_cyclic,cmap=cmap,levels=np.arange(bound_min,bound_max+.1,.1),extend=extend,transform=ccrs.PlateCarree())
    ax2.scatter(proxy_lons,proxy_lats,20,c='k',marker='o',alpha=1,transform=ccrs.PlateCarree())
    colorbar = plt.colorbar(map1,orientation='horizontal',ax=ax2,fraction=0.08,pad=0.02)
    colorbar.set_label(map_value_name,fontsize=16)
    colorbar.ax.tick_params(labelsize=16)
    ax2.coastlines()
    ax2.gridlines(color='k',linestyle='--',draw_labels=False)
    ax2.set_title('(b) Spatial '+map_value_name.lower()+'s between model and reconstruction (global mean='+str('{:.2f}'.format(metric_global[var_name][map_value_name]))+')',fontsize=20,loc='left')
    #
    plt.tight_layout()
    plt.suptitle('Pseudoproxy experiment',fontsize=26)
    plt.subplots_adjust(top=.93)
    #
    if save_instead_of_plot:
        plt.savefig('figures/PaperFigA1_pseudoproxy_'+var_name+'_and_'+map_value_name+'_map.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    #
    #
    # For each millennium, plot the difference between the reconstruction and model
    age_begin = 9000; i = 0
    for i,age_begin in enumerate(np.arange(0,11001,1000)):
        age_end = age_begin + 1000
        #
        # Plot the main composite
        plt.figure(figsize=(16,15))
        ax1 = plt.subplot2grid((3,1),(0,0))
        correlation_gmt = np.corrcoef(np.mean(recon_global[var_name],axis=1),model_data[var_name+'_gmt_selected'])[0,1]
        ax1.fill_between(ages_da,np.percentile(recon_global[var_name],2.5,axis=1),np.percentile(recon_global[var_name],97.5,axis=1),color='tab:blue',alpha=0.5)
        ax1.plot(ages_da,np.mean(recon_global[var_name],axis=1),'tab:blue',label='Reconstruction, mean and 95% range')
        ax1.plot(age_model,model_data[var_name+'_gmt'],'k',label='Model target ('+model_to_use_txt+')')
        ax1.legend(loc='lower right',fontsize=16)
        ax1.axvspan(age_begin,age_end,facecolor='gray',alpha=0.5)
        ax1.set_xlim(ages_da[-1],ages_da[0])
        ax1.set_ylabel(yaxis_txt[var_name]+' ('+unit_txt[var_name]+')',fontsize=16)
        ax1.set_xlabel('Age (B.P.)',fontsize=16)
        ax1.set_title('(a) Global-mean '+title_txt[var_name]+' ('+unit_txt[var_name]+', R='+str('{:.2f}'.format(correlation_gmt))+')',fontsize=20,loc='left')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        #
        # Make a map of values at every point
        ax2 = plt.subplot2grid((3,1),(1,0),rowspan=2,projection=ccrs.Robinson(central_longitude=0))
        ind_selected = np.where((ages_da >= age_begin) & (ages_da < age_end))[0]
        model_map = np.mean(model_data[var_name+'_selected'][ind_selected,:,:],axis=0)
        recon_map = np.mean(recon_spatial[var_name][ind_selected,:,:],axis=0)
        diff_map = recon_map - model_map
        diff_map_cyclic,lon_cyclic = cutil.add_cyclic_point(diff_map,coord=lon)
        #map1 = ax2.contourf(lon_cyclic,lat,diff_map_cyclic,cmap='bwr',levels=np.arange(-2,2.1,.1),extend='both',transform=ccrs.PlateCarree())
        map1 = ax2.contourf(lon_cyclic,lat,diff_map_cyclic,cmap='bwr',levels=np.arange(-5,5.1,.5),extend='both',transform=ccrs.PlateCarree())
        ax2.scatter(proxy_lons,proxy_lats,20,c='k',marker='o',alpha=1,transform=ccrs.PlateCarree())
        colorbar = plt.colorbar(map1,orientation='horizontal',ax=ax2,fraction=0.08,pad=0.02)
        colorbar.set_label(map_value_name,fontsize=16)
        colorbar.ax.tick_params(labelsize=16)
        ax2.coastlines()
        ax2.gridlines(color='k',linestyle='--',draw_labels=False)
        ax2.set_title('(b) Difference for reconstruction - model for ages '+str(age_begin)+'-'+str(age_begin)+' yr BP',fontsize=20,loc='left')
        #
        plt.tight_layout()
        plt.suptitle('Pseudoproxy experiment',fontsize=26)
        plt.subplots_adjust(top=.93)
        #
        if save_instead_of_plot:
            plt.savefig('figures/pseudoproxy_diff_'+str(12-i).zfill(2)+'_'+var_name+'_and_'+map_value_name+'_map.png',dpi=300,format='png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()


#%%
# Plot another comparison
plt.figure(figsize=(16,15))
ax1 = plt.subplot2grid((3,1),(0,0))
ax1.plot(ages_da,np.nanmean(recon_mean_proxies,axis=1),'tab:blue',label='Reconstructed pseudoproxies')
ax1.plot(ages_da,np.nanmean(proxy_values,axis=1),      'k',       label='Pseudoproxies')
ax1.legend(loc='lower right',fontsize=16)
#ax1.set_ylim(-2,1)
ax1.set_xlim(ages_da[-1],ages_da[0])
ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=16)
ax1.set_xlabel('Age (B.P.)',fontsize=16)
ax1.set_title('Mean of pseudoproxies vs mean of reconstructed pseudoproxies',fontsize=20,loc='left')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

if save_instead_of_plot:
    plt.savefig('figures/pseudoproxy_means.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%% OUTPUT
if make_output:
    #
    output_filename = 'output_'+recon_filename
    print('Saving values as '+output_filename)
    #
    # Save all data into a netCDF file
    output_dir = '/home/mpe32/analysis/15_Holocene_Reconstruction/analyze_results/06_pseudoproxy_experiments/data/'
    outputfile = netCDF4.Dataset(output_dir+output_filename,'w')
    outputfile.createDimension('lat',  n_lat)
    outputfile.createDimension('lon',  n_lon)
    outputfile.createDimension('proxy',n_proxies)
    #
    output_corr  = outputfile.createVariable('tas_corr',      'f4',('lat','lon',))
    output_error = outputfile.createVariable('tas_mean_error','f4',('lat','lon',))
    output_rmse  = outputfile.createVariable('tas_rmse',      'f4',('lat','lon',))
    output_ce    = outputfile.createVariable('tas_ce',        'f4',('lat','lon',))
    output_lat   = outputfile.createVariable('lat',           'f4',('lat',))
    output_lon   = outputfile.createVariable('lon',           'f4',('lon',))
    output_proxy_lats = outputfile.createVariable('proxy_lats','f4',('proxy',))
    output_proxy_lons = outputfile.createVariable('proxy_lons','f4',('proxy',))
    #
    output_corr[:]  = gridded_metric['tas']['Correlation']
    output_error[:] = gridded_metric['tas']['Mean_error']
    output_rmse[:]  = gridded_metric['tas']['RMSE']
    output_ce[:]    = gridded_metric['tas']['CE']
    output_lat[:]   = lat
    output_lon[:]   = lon
    output_proxy_lats[:] = proxy_lats
    output_proxy_lons[:] = proxy_lons
    #
    outputfile.close()
    print(' === DONE ===')
    sys.exit()

