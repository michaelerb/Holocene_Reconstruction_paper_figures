#==============================================================================
# Analyze output from the Holocene DA. This makes calculations for Tables B1
# and B3 in the paper
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
import pickle

make_output          = False
make_figures         = False
save_instead_of_plot = False
plt.style.use('ggplot')
output_dir = '/home/mpe32/analysis/15_Holocene_Reconstruction/analyze_results/paper_figures/data/'

# Select the variable to examine
var_name = 'tas'

# Pick the reconstructions to analyze
filenames = {}

# Experiments for Table 2
filenames['Exp_01'] = 'holocene_recon_2022-04-04_16:55:44.482606_pseudo_default.nc'
filenames['Exp_02'] = 'holocene_recon_2022-04-04_17:24:25.021686_pseudo_basicgrid.nc'
filenames['Exp_03'] = 'holocene_recon_2022-04-04_16:57:38.485355_pseudo_goodphysics.nc'
filenames['Exp_04'] = 'holocene_recon_2022-04-04_17:24:11.800109_pseudo_basicgrid_goodphysics.nc'
filenames['Exp_05'] = 'holocene_recon_2022-04-04_17:04:42.441901_pseudo_prior_constant_mean.nc'
filenames['Exp_06'] = 'holocene_recon_2022-04-04_17:19:45.659888_pseudo_prior_constant.nc'
filenames['Exp_07'] = 'holocene_recon_2022-04-05_01:40:10.392185_pseudo_locrad_25k.nc'
filenames['Exp_08'] = 'holocene_recon_2022-04-04_17:09:47.364905_pseudo_prior_3010.nc'

# Extra prior window experiments
filenames['window_1010yr'] = 'holocene_recon_2022-04-04_17:13:09.999569_pseudo_prior_1010.nc'
filenames['window_3010yr'] = 'holocene_recon_2022-04-04_17:09:47.364905_pseudo_prior_3010.nc'
filenames['window_4010yr'] = 'holocene_recon_2022-04-04_17:02:42.356825_pseudo_prior_4010.nc'
filenames['window_5010yr'] = 'holocene_recon_2022-04-04_16:55:44.482606_pseudo_default.nc'
filenames['window_6010yr'] = 'holocene_recon_2022-04-04_17:06:59.771529_pseudo_prior_6010.nc'
filenames['window_7010yr'] = 'holocene_recon_2022-04-04_17:09:51.890160_pseudo_prior_7010.nc'
filenames['window_9010yr'] = 'holocene_recon_2022-04-04_17:15:51.842729_pseudo_prior_9010.nc'

# Extra locrad experiments
filenames['locrad1_none'] = 'holocene_recon_2022-04-04_16:55:44.482606_pseudo_default.nc'
filenames['locrad2_25K']  = 'holocene_recon_2022-04-05_01:40:10.392185_pseudo_locrad_25k.nc'
filenames['locrad3_20K']  = 'holocene_recon_2022-04-05_01:40:13.403417_pseudo_locrad_20k.nc'
filenames['locrad4_15K']  = 'holocene_recon_2022-04-05_01:03:09.975653_pseudo_locrad_15k.nc'

# Experiments for Table 3
filenames['trace_recon_hadcm3']  = 'holocene_recon_2022-04-04_17:03:44.263613_pseudo_trace_recon_hadcm3.nc'
filenames['famous_recon_hadcm3'] = 'holocene_recon_2022-04-05_13:08:59.983430_pseudo_famous_recon_hadcm3.nc'
filenames['2model_recon_hadcm3'] = 'holocene_recon_2022-04-05_13:19:31.311532_pseudo_2model_recon_hadcm3.nc'

filenames['hadcm3_recon_trace']  = 'holocene_recon_2022-04-04_16:55:44.482606_pseudo_default.nc'
filenames['famous_recon_trace']  = 'holocene_recon_2022-04-05_12:58:39.732295_pseudo_famous_recon_trace.nc'
filenames['2model_recon_trace']  = 'holocene_recon_2022-04-05_13:05:55.034482_pseudo_2model_recon_trace.nc'

filenames['hadcm3_recon_famous'] = 'holocene_recon_2022-04-05_13:07:54.106920_pseudo_hadcm3_recon_famous.nc'
filenames['trace_recon_famous']  = 'holocene_recon_2022-04-05_12:58:32.660003_pseudo_trace_recon_famous.nc'
filenames['2model_recon_famous'] = 'holocene_recon_2022-04-05_13:28:00.829590_pseudo_2model_recon_famous.nc'

"""
# Experiments with constant priors
filenames['trace_recon_hadcm3_cp']  = 'holocene_recon_2022-04-05_16:23:42.258431_pseudo_trace_recon_hadcm3_constantprior.nc'
filenames['famous_recon_hadcm3_cp'] = 'holocene_recon_2022-04-05_16:40:00.936213_pseudo_famous_recon_hadcm3_constantprior.nc'
filenames['2model_recon_hadcm3_cp'] = 'holocene_recon_2022-04-05_18:02:46.708682_pseudo_2model_recon_hadcm3_constantprior.nc'

filenames['hadcm3_recon_trace_cp']  = 'holocene_recon_2022-04-04_17:19:45.659888_pseudo_prior_constant.nc'
filenames['famous_recon_trace_cp']  = 'holocene_recon_2022-04-05_17:49:51.954200_pseudo_famous_recon_trace_constantprior.nc'
filenames['2model_recon_trace_cp']  = 'holocene_recon_2022-04-05_18:04:57.031010_pseudo_2model_recon_trace_constantprior.nc'

filenames['hadcm3_recon_famous_cp'] = 'holocene_recon_2022-04-05_17:45:44.608440_pseudo_hadcm3_recon_famous_constantprior.nc'
filenames['trace_recon_famous_cp']  = 'holocene_recon_2022-04-05_16:23:59.540477_pseudo_trace_recon_famous_constantprior.nc'
filenames['2model_recon_famous_cp'] = 'holocene_recon_2022-04-05_16:41:35.258423_pseudo_2model_recon_famous_constantprior.nc'
"""

#%%
# Loop through each experiment, making calculations
keys = list(filenames.keys())

# Set up some arrays, for saving values later
n_keys = len(keys)
values_all = {}
metrics_to_save = ['R_gmt','CE_gmt','RMSE_gmt','R_spatialmean','CE_spatialmean','RMSE_spatialmean','R_proxies','CE_proxies','RMSE_proxies','R_prior','CE_prior','RMSE_prior','R_diff','CE_diff','RMSE_diff']
for metric in metrics_to_save:
    values_all[metric] = np.zeros((n_keys)); values_all[metric][:] = np.nan

recon_global,prior_global,recon_spatial = {},{},{}
#counter = 0; key = keys[counter]
for counter,key in enumerate(keys):
    #
    print(' === Calculating '+str(counter+1)+'/'+str(n_keys)+': '+key+' ===')
    #
    # Load the Holocene reconstruction
    recon_filename = filenames[key]
    recon_dir = '/projects/pd_lab/data/data_assimilation/'
    handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
    recon_global[var_name]  = handle['recon_'+var_name+'_global_mean'].values
    prior_global[var_name]  = handle['prior_'+var_name+'_global_mean'].values
    recon_spatial[var_name] = handle['recon_'+var_name+'_mean'].values
    proxy_values            = handle['proxy_values'].values
    recon_mean_proxies      = handle['proxyrecon_mean'].values
    prior_mean_proxies      = handle['proxyprior_mean'].values
    proxy_uncertainty       = handle['proxy_uncertainty'].values
    proxies_assimilated_all = handle['proxies_assimilated'].values
    proxy_metadata = handle['proxy_metadata'].values
    ages_da        = handle['ages'].values
    lat            = handle['lat'].values
    lon            = handle['lon'].values
    options_da     = handle['options'].values
    handle.close()
    #
    # Get the options
    options = {}
    keys_to_ints_list = ['age_range_to_reconstruct','reference_period','age_range_model']
    for i in range(len(options_da)):
        text,value = options_da[i].split(':')
        if '[' in value: value = value.replace('[','').replace(']','').replace("'",'').replace(' ','').split(',')
        if text in keys_to_ints_list: value = [int(i) for i in value]
        if (isinstance(value,str)) and (value.isdecimal()): value = int(value)
        options[text] = value
    #
    # Get information about the pseudoproxy file
    proxies_to_use = options['proxy_datasets_to_assimilate'][0].split('_')[1]
    model_to_use   = options['proxy_datasets_to_assimilate'][0].split('_')[3]
    noise_to_use   = options['proxy_datasets_to_assimilate'][0].split('_')[5]
    #time_res = options['time_resolution']
    #age_range_model = str(options['age_range_model'][1])+'-'+str(options['age_range_model'][0])
    #
    # Load the model data
    model_data = {}
    model_dir = '/projects/pd_lab/data/data_assimilation/models/processed_model_data/'
    model_filename = model_to_use+'_regrid.21999-0BP.'+var_name+'.timeres_10.nc'
    handle_model = xr.open_dataset(model_dir+model_filename,decode_times=False)
    model_data[var_name] = handle_model[var_name].values
    lat_model        = handle_model['lat'].values
    lon_model        = handle_model['lon'].values
    age_model        = handle_model['age'].values
    time_ndays_model = handle_model['days_per_month_all'].values
    handle_model.close()
    model_data[var_name] = np.flip(model_data[var_name],axis=0)
    #
    # Flip the data so that it runs from recent to old
    age_model        = np.flip(age_model,       axis=0)
    time_ndays_model = np.flip(time_ndays_model,axis=0)
    #
    #
    ### CALCULATIONS
    #
    # Set some text
    title_txt     = {'tas':'temperature','precip':'precipitation'}
    unit_txt      = {'tas':'$^\circ$C',  'precip':'$mm day^{-1}$'}
    yaxis_txt     = {'tas':'$\Delta$T',  'precip':'$\Delta$P'}
    anomaly_value = {'tas':2,            'precip':1}
    cmap_selected = {'tas':'bwr',        'precip':'BrBG'}
    #
    # Get some dimensions
    model_res = 10
    n_ages = proxy_values.shape[0]
    n_proxies = proxy_values.shape[1]
    n_lat     = len(lat)
    n_lon     = len(lon)
    #
    proxy_names = proxy_metadata[:,0]
    proxy_tsids = proxy_metadata[:,1]
    proxy_lats  = proxy_metadata[:,2].astype(float)
    proxy_lons  = proxy_metadata[:,3].astype(float)
    if n_proxies != len(proxy_lats): print('Different numbers of proxies between original file and metadata:',n_proxies,len(proxy_lats))
    #
    #
    ###
    #
    # Figure out which proxies were assimilated
    ind_proxies_assimilated = np.where(np.mean(proxies_assimilated_all,axis=0) > 0)[0]
    ind_proxies_leftout     = np.where(np.mean(proxies_assimilated_all,axis=0) == 0)[0]
    n_proxies_assim = len(ind_proxies_assimilated)
    #
    print('Number of proxies...')
    print('Assimilated:',len(ind_proxies_assimilated))
    print('Omitted:    ',len(ind_proxies_leftout))
    #
    # Function to calculate metrics for the proxies.
    proxy_data_all,proxy_recon_all = proxy_values,recon_mean_proxies
    proxy_data_all,proxy_recon_all = proxy_values,prior_mean_proxies
    def calc_metrics(proxy_data_all,proxy_recon_all):
        #
        # Set up arrays
        correlations_for_proxies = np.zeros((n_proxies)); correlations_for_proxies[:] = np.nan
        mean_errors_for_proxies  = np.zeros((n_proxies)); mean_errors_for_proxies[:]  = np.nan
        RMSE_for_proxies         = np.zeros((n_proxies)); RMSE_for_proxies[:]         = np.nan
        CE_for_proxies           = np.zeros((n_proxies)); CE_for_proxies[:]           = np.nan
        #
        for i in range(n_proxies):
            #
            #print('Proxy '+str(i)+'/'+str(n_proxies))
            #
            # Get data for proxy and reconstruction
            proxy_data  = proxy_data_all[:,i]
            proxy_recon = proxy_recon_all[:,i]
            #
            # Find ages which have data in both datasets
            valid_indices = np.isfinite(proxy_recon)*np.isfinite(proxy_data)
            #
            # Compute correlations, mean errors, RMSE, and coeffients of efficiency (CE)
            correlations_for_proxies[i] = np.corrcoef(proxy_recon[valid_indices],proxy_data[valid_indices])[0,1]
            mean_errors_for_proxies[i]  = np.nanmean(np.abs(proxy_recon-proxy_data))
            RMSE_for_proxies[i]         = np.sqrt(np.nanmean(np.square(proxy_recon-proxy_data)))
            CE_for_proxies[i]           = 1 - ( np.sum(np.power(proxy_recon[valid_indices]-proxy_data[valid_indices],2),axis=0) / np.sum(np.power(proxy_data[valid_indices]-np.mean(proxy_data[valid_indices],axis=0),2),axis=0) )
        #
        return correlations_for_proxies,mean_errors_for_proxies,RMSE_for_proxies,CE_for_proxies
    #
    correlations_for_proxies,mean_errors_for_proxies,RMSE_for_proxies,CE_for_proxies = calc_metrics(proxy_values,recon_mean_proxies)
    correlations_for_prior,  mean_errors_for_prior,  RMSE_for_prior,  CE_for_prior   = calc_metrics(proxy_values,prior_mean_proxies)
    #
    # Calculate the differences
    correlations_diff = correlations_for_proxies - correlations_for_prior
    mean_errors_diff  = mean_errors_for_proxies  - mean_errors_for_prior
    RMSE_diff         = RMSE_for_proxies         - RMSE_for_prior
    CE_diff           = CE_for_proxies           - CE_for_prior
    #
    ###
    #
    # Remove the mean of the reference period
    indices_refperiod_recon = np.where((ages_da   >= options['reference_period'][0]) & (ages_da   < options['reference_period'][1]))[0]
    indices_refperiod_model = np.where((age_model >= options['reference_period'][0]) & (age_model < options['reference_period'][1]))[0]
    #
    gridded_metric,metric_global = {},{}
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
    #
    #
    ### FIGURES
    #
    filename_txt = recon_filename.split('timevarying_')[-1].split('.nc')[0]
    #
    # Plot the main composite
    for value_name in ['Correlation','CE','RMSE']:
        #
        gmt_recon = np.mean(recon_global[var_name],axis=1)
        gmt_model = model_data[var_name+'_gmt_selected']
        correlation_gmt = np.corrcoef(gmt_recon,gmt_model)[0,1]
        CE_gmt = 1 - ( np.sum(np.power(gmt_recon-gmt_model,2),axis=0) / np.sum(np.power(gmt_model-np.mean(gmt_model,axis=0),2),axis=0) )
        RMSE_gmt = np.sqrt(np.nanmean(np.square(gmt_recon-gmt_model)))
        #
        if make_figures:
            #
            plt.figure(figsize=(16,15))
            ax1 = plt.subplot2grid((3,1),(0,0))
            ax2 = plt.subplot2grid((3,1),(1,0),rowspan=2,projection=ccrs.Robinson(central_longitude=0))
            #
            ax1.fill_between(ages_da,np.percentile(prior_global[var_name],2.5,axis=1),np.percentile(prior_global[var_name],97.5,axis=1),color='gray',alpha=0.25)
            ax1.plot(ages_da,np.mean(prior_global[var_name],axis=1),'gray',label='Prior, mean and 95% range')
            ax1.fill_between(ages_da,np.percentile(recon_global[var_name],2.5,axis=1),np.percentile(recon_global[var_name],97.5,axis=1),color='tab:blue',alpha=0.5)
            ax1.plot(ages_da,np.mean(recon_global[var_name],axis=1),'tab:blue',label='Reconstruction, mean and 95% range')
            ax1.plot(age_model,model_data[var_name+'_gmt'],'k',label='Model')
            ax1.legend(loc='lower right')
            ax1.set_ylabel(yaxis_txt[var_name]+' ('+unit_txt[var_name]+')')
            #
            ax1.set_xlim(ages_da[-1],ages_da[0])
            ax1.set_xlabel('Age (B.P.)')
            ax1.set_title('(a) '+key+', global-mean '+title_txt[var_name]+' ('+unit_txt[var_name]+'),  R='+str('{:.2f}'.format(correlation_gmt))+'  |  CE='+str('{:.2f}'.format(CE_gmt))+'  |  RMSE='+str('{:.2f}'.format(RMSE_gmt)),fontsize=18,loc='left')
            #
            # Make a map of values at every point
            if   value_name == 'Correlation': bound_min = -1; bound_max = 1; cmap = 'Spectral'; extend = 'neither'
            elif value_name == 'CE':          bound_min = -1; bound_max = 1; cmap = 'Spectral'; extend = 'both'
            elif value_name == 'RMSE':        bound_min = 0;  bound_max = 2; cmap = 'viridis';  extend = 'max'
            gridded_metric_cyclic,lon_cyclic = cutil.add_cyclic_point(gridded_metric[var_name][value_name],coord=lon)
            map1 = ax2.contourf(lon_cyclic,lat,gridded_metric_cyclic,cmap=cmap,levels=np.arange(bound_min,bound_max+.1,.1),extend=extend,transform=ccrs.PlateCarree())
            ax2.scatter(proxy_lons,proxy_lats,20,c='k',marker='o',alpha=1,transform=ccrs.PlateCarree())
            plt.colorbar(map1,orientation='horizontal',label=value_name,ax=ax2,fraction=0.08,pad=0.02)
            ax2.coastlines()
            ax2.gridlines(color='k',linestyle='--',draw_labels=False)
            ax2.set_title('(b) '+value_name+' between model and reconstruction at every point (global mean='+str('{:.2f}'.format(metric_global[var_name][value_name]))+')',fontsize=18,loc='left')
            #
            if save_instead_of_plot:
                #plt.savefig('figures/pseudo_'+var_name+'_'+value_name+'_map_'+key+'_'+filename_txt+'.png',dpi=150,format='png',bbox_inches='tight')
                plt.savefig('figures/pseudo_'+var_name+'_'+value_name+'_map_'+key+'.png',dpi=150,format='png',bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    #
    # SAVE VALUES FOR TABLE 2
    values_all['R_gmt'][counter]    = correlation_gmt
    values_all['CE_gmt'][counter]   = CE_gmt
    values_all['RMSE_gmt'][counter] = RMSE_gmt
    values_all['R_spatialmean'][counter]    = metric_global[var_name]['Correlation']
    values_all['CE_spatialmean'][counter]   = metric_global[var_name]['CE']
    values_all['RMSE_spatialmean'][counter] = metric_global[var_name]['RMSE']
    #
    # SAVE VALUES FOR TABLE 3
    # Note: when using a contant prior, it doesn't make much sense to calculate correlations with the prior
    # It's because the prior is almost perfectly flat (there's just some noise), and sometimes perfectly flat, resulting in nans.
    values_all['R_proxies'][counter]    = np.nanmedian(correlations_for_proxies[ind_proxies_assimilated])
    values_all['CE_proxies'][counter]   = np.nanmedian(CE_for_proxies[ind_proxies_assimilated])
    values_all['RMSE_proxies'][counter] = np.nanmedian(RMSE_for_proxies[ind_proxies_assimilated])
    values_all['R_prior'][counter]      = np.nanmedian(correlations_for_prior[ind_proxies_assimilated])
    values_all['CE_prior'][counter]     = np.nanmedian(CE_for_prior[ind_proxies_assimilated])
    values_all['RMSE_prior'][counter]   = np.nanmedian(RMSE_for_prior[ind_proxies_assimilated])
    values_all['R_diff'][counter]       = np.nanmedian(correlations_diff[ind_proxies_assimilated])
    values_all['CE_diff'][counter]      = np.nanmedian(CE_diff[ind_proxies_assimilated])
    values_all['RMSE_diff'][counter]    = np.nanmedian(RMSE_diff[ind_proxies_assimilated])
    #
    ### OUTPUT
    if make_output:
        #
        output_filename = 'output_'+recon_filename
        print('Saving values as '+output_filename)
        #
        # Save all data into a netCDF file
        outputfile = netCDF4.Dataset(output_dir+output_filename,'w')
        outputfile.createDimension('lat',  n_lat)
        outputfile.createDimension('lon',  n_lon)
        outputfile.createDimension('proxy',n_proxies_assim)
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
        output_proxy_lats[:] = proxy_lats[ind_proxies_assimilated]
        output_proxy_lons[:] = proxy_lons[ind_proxies_assimilated]
        #
        outputfile.close()
        print(' === Finished making output file ===')


#%%
# Calculate the ranks of each metric
ranks_all = {}
for metric in metrics_to_save:
    ind_sorted = np.argsort(values_all[metric])
    ranks = np.empty_like(ind_sorted)
    ranks[ind_sorted] = np.arange(n_keys,0,-1)
    ranks_all[metric] = ranks


#%% TABLE B1 and B3 VALUES
# Note: this script outputs more values than are included in the paper tables.

# Create a text file to save into
output_file_table = open(output_dir+'Table2_3_values.txt','w')
column_labels_table = '           Exp, R_GMT, CE_GMT, RMSE_GMT, R_spatialmean, CE_spatialmean, RMSE_spatialmean'
output_file_table.write(column_labels_table+'\n')
print_format_table     = '%26s, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f %1s'
print_format_table_ind = '%26s, %5s, %5s, %5s, %5s, %5s, %5s %1s'

# Print and save the values
print(' === VALUES FOR TABLE 2 AND/OR 3 ===')
print(column_labels_table)
for i,key in enumerate(keys):
    print(print_format_table                   % (key,values_all['R_gmt'][i],values_all['CE_gmt'][i],values_all['RMSE_gmt'][i],values_all['R_spatialmean'][i],values_all['CE_spatialmean'][i],values_all['RMSE_spatialmean'][i],''))
    output_file_table.write(print_format_table % (key,values_all['R_gmt'][i],values_all['CE_gmt'][i],values_all['RMSE_gmt'][i],values_all['R_spatialmean'][i],values_all['CE_spatialmean'][i],values_all['RMSE_spatialmean'][i],'\n'))

print(' === RANKS ===')
print(column_labels_table)
for i,key in enumerate(keys):
    print(print_format_table_ind                   % (key,ranks_all['R_gmt'][i],ranks_all['CE_gmt'][i],ranks_all['RMSE_gmt'][i],ranks_all['R_spatialmean'][i],ranks_all['CE_spatialmean'][i],ranks_all['RMSE_spatialmean'][i],''))
    output_file_table.write(print_format_table_ind % (key,ranks_all['R_gmt'][i],ranks_all['CE_gmt'][i],ranks_all['RMSE_gmt'][i],ranks_all['R_spatialmean'][i],ranks_all['CE_spatialmean'][i],ranks_all['RMSE_spatialmean'][i],'\n'))

# Close the output text file
output_file_table.close()


#%% ADDITIONAL VALUES

# Create a text file to save into
output_file_more = open(output_dir+'values_more.txt','w')
column_labels_more = '           Exp, R_proxies, CE_proxies, RMSE_proxies, R_prior, CE_prior, RMSE_prior, R_diff, CE_diff, RMSE_diff'
output_file_more.write(column_labels_more+'\n')
print_format_more     = '%26s, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f %1s'
print_format_more_ind = '%26s, %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s %1s'

# Print and save the values
print(' === MORE VALUES ===')
print(column_labels_more)
for i,key in enumerate(keys):
    print(print_format_more                  % (key,values_all['R_proxies'][i],values_all['CE_proxies'][i],values_all['RMSE_proxies'][i],values_all['R_prior'][i],values_all['CE_prior'][i],values_all['RMSE_prior'][i],values_all['R_diff'][i],values_all['CE_diff'][i],values_all['RMSE_diff'][i],''))
    output_file_more.write(print_format_more % (key,values_all['R_proxies'][i],values_all['CE_proxies'][i],values_all['RMSE_proxies'][i],values_all['R_prior'][i],values_all['CE_prior'][i],values_all['RMSE_prior'][i],values_all['R_diff'][i],values_all['CE_diff'][i],values_all['RMSE_diff'][i],'\n'))

# Print and save the values
print(' === MORE VALUES ===')
print(column_labels_more)
for i,key in enumerate(keys):
    print(print_format_more_ind                  % (key,ranks_all['R_proxies'][i],ranks_all['CE_proxies'][i],ranks_all['RMSE_proxies'][i],ranks_all['R_prior'][i],ranks_all['CE_prior'][i],ranks_all['RMSE_prior'][i],ranks_all['R_diff'][i],ranks_all['CE_diff'][i],ranks_all['RMSE_diff'][i],''))
    output_file_more.write(print_format_more_ind % (key,ranks_all['R_proxies'][i],ranks_all['CE_proxies'][i],ranks_all['RMSE_proxies'][i],ranks_all['R_prior'][i],ranks_all['CE_prior'][i],ranks_all['RMSE_prior'][i],ranks_all['R_diff'][i],ranks_all['CE_diff'][i],ranks_all['RMSE_diff'][i],'\n'))

# Close the output text file
output_file_more.close()


#%%
# Save values to a pickle file
pickle_file = open(output_dir+'pseudo_metrics.pkl','wb')
pickle.dump(values_all,pickle_file)
pickle_file.close()
