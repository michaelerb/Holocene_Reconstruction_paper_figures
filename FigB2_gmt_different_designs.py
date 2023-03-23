#=============================================================================
# This script compares global mean temperatures from multiple reconstructions.
# The reconstructions have different settings.  It makes Fig. B3 for the
# paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/data_assimilation')
import da_utils
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

save_instead_of_plot = False


#%% LOAD DATA

# Pick the reconstruction to analyze
filenames = {}; colors = {}
filenames['Default']               = 'holocene_reconstruction.nc'
filenames['25K locrad']            = 'holocene_recon_2022-03-30_03:27:58.939982_locrad_25k.nc'
filenames['20K locrad']            = 'holocene_recon_2022-03-30_11:54:25.273718_locrad_20k.nc'
filenames['15K locrad']            = 'holocene_recon_2022-03-30_09:58:56.392223_locrad_15k.nc'
filenames['20% R']                 = 'holocene_recon_2022-03-30_09:52:27.293540_20percent_r.nc'
filenames['HadCM3 prior']          = 'holocene_recon_2022-03-30_09:38:00.219634_prior_hadcm3.nc'
filenames['TraCE-21k prior']       = 'holocene_recon_2022-03-30_09:37:23.582231_prior_trace.nc'
filenames['200yr res']             = 'holocene_recon_2022-03-30_09:09:27.307967_200yr.nc'
filenames['200yr res, 25K locrad'] = 'holocene_recon_2022-03-30_09:09:06.139690_200yr_locrad_25k.nc'

#experiments = list(filenames.keys())
experiments = ['Default',
               '25K locrad',
               '20K locrad',
               '15K locrad',
               '20% R',
               'HadCM3 prior',
               'TraCE-21k prior',
               '200yr res',
               '200yr res, 25K locrad']

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+filenames['Default'],decode_times=False)
lat     = handle['lat'].values
lon     = handle['lon'].values
handle.close()

recon_spatial,recon_gmt,proxy_uncertainty,ages_da,prior_gmt = {},{},{},{},{}
for exp in experiments:
    print('Loading data:',exp)
    handle = xr.open_dataset(recon_dir+'results/'+filenames[exp],decode_times=False)
    recon_spatial[exp]     = handle['recon_tas_mean'].values
    recon_gmt[exp]         = handle['recon_tas_global_mean'].values
    prior_gmt[exp]         = handle['prior_tas_global_mean'].values
    proxy_uncertainty[exp] = handle['proxy_uncertainty'].values
    ages_da[exp]           = handle['ages'].values
    handle.close()


#%% CALCULATIONS

# Regrid the default experiment to 200 year resolution
n_ages = recon_spatial['Default'].shape[0]
n_lat  = recon_spatial['Default'].shape[1]
n_lon  = recon_spatial['Default'].shape[2]
ages_da['Default_200yr'] = np.mean(np.reshape(ages_da['Default'],(int(n_ages/20),20)),axis=1)
recon_spatial['Default_200yr'] = np.mean(np.reshape(recon_spatial['Default'],(int(n_ages/20),20,n_lat,n_lon)),axis=1)

# Print the GMT variance of different experiments
print(' === GMT variance ===')
gmt_200yr = da_utils.global_mean(recon_spatial['Default_200yr'],lat,1,2)
print('Default:',np.var(np.mean(recon_gmt['Default'],axis=1)))
print('Default 200yr:',np.var(gmt_200yr))
print('200yr:',np.var(np.mean(recon_gmt['200yr res'],axis=1)))


#%%
# Compute correlations, mean errors, RMSE, and coeffients of efficiency (CE)
def calc_metrics(ts1,ts2):
    #
    valid_indices = np.isfinite(ts1)*np.isfinite(ts2)
    ts1_valid = ts1[valid_indices]
    ts2_valid = ts2[valid_indices]
    #
    correlation = np.corrcoef(ts1_valid,ts2_valid)[0,1]
    mean_error  = np.nanmean(np.abs(ts1_valid-ts2_valid))
    RMSE        = np.sqrt(np.nanmean(np.square(ts1_valid-ts2_valid)))
    CE_1        = 1 - ( np.sum(np.power(ts1_valid-ts2_valid,2),axis=0) / np.sum(np.power(ts2_valid-np.mean(ts2_valid,axis=0),2),axis=0) )
    CE_2        = 1 - ( np.sum(np.power(ts2_valid-ts1_valid,2),axis=0) / np.sum(np.power(ts1_valid-np.mean(ts1_valid,axis=0),2),axis=0) )
    CE_mean = (CE_1 + CE_2)/2
    #
    return correlation,mean_error,RMSE,CE_mean

# Compute correlation and other metrics between the reconstruction and the model data at every gridpoint
n_lat = len(lat)
n_lon = len(lon)
metrics,metrics_global = {},{}
for exp in experiments:
    print('Calculating metrics: '+exp)
    metrics[exp] = {}
    metrics[exp]['Correlation'] = np.zeros((n_lat,n_lon)); metrics[exp]['Correlation'][:] = np.nan
    metrics[exp]['RMSE']        = np.zeros((n_lat,n_lon)); metrics[exp]['RMSE'][:]        = np.nan
    metrics[exp]['CE']          = np.zeros((n_lat,n_lon)); metrics[exp]['CE'][:]          = np.nan
    for j in range(n_lat):
        for i in range(n_lon):
            ts1 = recon_spatial[exp][:,j,i]
            ts2 = recon_spatial['Default'][:,j,i]
            ts3 = recon_spatial['Default_200yr'][:,j,i]
            if '200yr' in exp: metrics[exp]['Correlation'][j,i],_,metrics[exp]['RMSE'][j,i],metrics[exp]['CE'][j,i] = calc_metrics(ts1,ts3)
            else:              metrics[exp]['Correlation'][j,i],_,metrics[exp]['RMSE'][j,i],metrics[exp]['CE'][j,i] = calc_metrics(ts1,ts2)
    #
    # Compute global-means of the metric
    metrics_global[exp] = {}
    for value_name in ['Correlation','RMSE','CE']:
        metrics_global[exp][value_name] = da_utils.global_mean(metrics[exp][value_name],lat,0,1)

# Remove the mean of the 0-1ka age interval from the reconstruction
"""
indices_ref_recon = np.where((ages_da >= 0) & (ages_da <= 1000))[0]
for exp in experiments: recon_gmt[exp] = recon_gmt[exp] - np.mean(recon_gmt[exp][indices_ref_recon,:])
"""

#%%
# Print the values
print('%50s: %4s, %4s, %4s' % ('Experiment','R','RMSE','CE'))
for i,exp in enumerate(experiments):
    print('%50s: %1.2f, %1.2f, %1.2f' % (exp,metrics_global[exp]['Correlation'],metrics_global[exp]['RMSE'],metrics_global[exp]['CE']))



#%%
# Compute the relative mid-Holocene temperature for each reconstruction
def calc_6ka_anom(data_ts,ages_ts,name_txt):
    #
    ind_anom = np.where((ages_ts >= 5500) & (ages_ts <= 6500))[0]
    ind_ref  = np.where((ages_ts >= 0)    & (ages_ts <= 1000))[0]
    anom_6ka = np.mean(data_ts[ind_anom]) - np.mean(data_ts[ind_ref])
    print(len(ind_anom),len(ind_ref))
    print(anom_6ka,name_txt)
    print('---')

for exp in experiments:
    calc_6ka_anom(np.mean(recon_gmt[exp],axis=1),ages_da[exp],exp)



#%% FIGURES
plt.style.use('ggplot')

# Plot time series of the reconstructions
f, ax1 = plt.subplots(1,1,figsize=(10,24))

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m']
offset = 1
for i,exp in enumerate(experiments):
    ax1.fill_between(ages_da[exp],np.nanmin(recon_gmt[exp],axis=1)-(i*offset),np.nanmax(recon_gmt[exp],axis=1)-(i*offset),color='tab:blue',alpha=0.25)
    ax1.plot(ages_da[exp],np.nanmean(recon_gmt[exp],axis=1)-(i*offset),color='tab:blue',linewidth=2,label=exp)
    ax1.axhline(y=(-i*offset),color='k',linewidth=1,linestyle='dashed',alpha=0.5)
    txt_position = [100,0.25-(i*offset)]; txt_align = 'right'
    if exp == 'Default':
        plt.text(txt_position[0],txt_position[1],'('+letters[i]+')'+' '+exp,fontsize=20,ha=txt_align)
    else:
        #plt.text(txt_position[0],txt_position[1],letters[i]+' '+exp+' (R='+str('{:.2f}'.format(metrics_global[exp]['Correlation']))+', CE='+str('{:.2f}'.format(metrics_global[exp]['CE']))+')',fontsize=20,ha=txt_align)
        plt.text(txt_position[0],txt_position[1],'('+letters[i]+')'+' '+exp,fontsize=20,ha=txt_align)
        ax1.plot(ages_da['Default'],np.nanmean(recon_gmt['Default'],axis=1)-(i*offset),'k-',linewidth=1)
    if 'varying mean' in exp:
        ax1.plot(ages_da[exp],np.nanmean(prior_gmt[exp],axis=1)-(i*offset),color='gray',linewidth=1)

label_txt = ['0','-0.25','-0.5','-0.75','-1','','','']
ax1.set_ylim(-10,.5)
ax1.set_yticks(np.arange(0,-9.8,-0.25))
ax1.set_yticklabels(label_txt*5)
ax1.tick_params(axis='both',which='major',labelsize=20)

ax2 = ax1.twinx()
ax2.set_ylim(-10,.5)
ax2.set_yticks(np.arange(-1,-8.8,-0.25))
ax2.set_yticklabels(label_txt*4)
ax2.tick_params(axis='both',which='major',labelsize=20)
ax2.grid(False)

ax1.set_xlabel('Age (yr BP)',fontsize=22)
ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=22)
ax1.set_xlim(12000,0)
ax1.set_title('Global-mean temperature anomalies',loc='center',fontsize=24)

if save_instead_of_plot:
    plt.savefig('figures/PaperFigB2_gmt_comparison.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


