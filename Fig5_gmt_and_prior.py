#==============================================================================
# Compare the global-mean reconstruction to the prior. This is Fig. 5 in the
# paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import copy

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
gmt_recon_all  = handle['recon_tas_global_mean'].values
gmt_prior_all  = handle['prior_tas_global_mean'].values
proxy_values   = handle['proxy_values'].values
proxy_metadata = handle['proxy_metadata'].values
ages_da        = handle['ages'].values
proxies_assimilated_all = handle['proxies_assimilated'].values
handle.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Count data coverage
proxy_values_assim = copy.deepcopy(proxy_values)
proxy_values_assim[proxies_assimilated_all == 0] = np.nan
data_counts_assim = np.sum(np.isfinite(proxy_values_assim),axis=1)
data_counts_assim_total = np.sum(np.sum(np.isfinite(proxy_values_assim),axis=0) > 0)

# Print the mean value at different times
ind_12ka = np.argmin(np.abs(ages_da - 12000))
ind_10ka = np.argmin(np.abs(ages_da - 10000))
gmt_recon_mean = np.nanmean(gmt_recon_all,axis=1)
print('Value at 12 ka:',gmt_recon_mean[ind_12ka])
print('Value at 10 ka:',gmt_recon_mean[ind_10ka])
print('Change for 10 - 12 ka:',gmt_recon_mean[ind_10ka]-gmt_recon_mean[ind_12ka])

# Print the timing of max warmth
ind_max = np.argmax(gmt_recon_mean)
print('Age of maximum temperature:',ages_da[ind_max])
print('Value of maximum temperature:',gmt_recon_mean[ind_max])



#%% FIGURES
plt.style.use('ggplot')

# Plot the main composite
f, (ax1,ax2) = plt.subplots(2,1,figsize=(15,10),sharex=True,gridspec_kw={'height_ratios': [2,1]})

ax1.fill_between(ages_da,np.nanmin(gmt_prior_all,axis=1),          np.nanmax(gmt_prior_all,axis=1),          color='gray',alpha=0.25)
ax1.fill_between(ages_da,np.nanpercentile(gmt_prior_all,16,axis=1),np.nanpercentile(gmt_prior_all,84,axis=1),color='gray',alpha=0.25)
ax1.plot(ages_da,np.nanmean(gmt_prior_all,axis=1),color='gray',    linewidth=3,label='Prior')
ax1.fill_between(ages_da,np.nanmin(gmt_recon_all,axis=1),          np.nanmax(gmt_recon_all,axis=1),          color='tab:blue',alpha=0.5)
ax1.fill_between(ages_da,np.nanpercentile(gmt_recon_all,16,axis=1),np.nanpercentile(gmt_recon_all,84,axis=1),color='tab:blue',alpha=0.5)
ax1.plot(ages_da,np.nanmean(gmt_recon_all,axis=1),color='tab:blue',linewidth=3,label='Reconstruction')
ax1.axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
ax1.legend(loc='lower right',ncol=2,fontsize=16)
ax1.set_xlim(12000,0)
ax1.set_ylim(-4,1)
ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=20)
ax1.tick_params(axis='both',which='major',labelsize=18)
ax1.set_title('(a) Global-mean temperature anomaly from reconstruction and prior ($^\circ$C)',fontsize=20,loc='left')

ax2.fill_between(ages_da,data_counts_assim*0,data_counts_assim,color='tab:blue',alpha=0.5,label='Assimilated (n='+str(data_counts_assim_total)+')')
ax2.legend(loc='upper left',fontsize=16)
ax2.set_ylim(ymin=0)
ax2.set_ylabel('# proxies',fontsize=20)
ax2.set_xlabel('Age (yr BP)',fontsize=20)
ax2.tick_params(axis='both',which='major',labelsize=20)
ax2.set_title('(b) Proxy data coverage',fontsize=20,loc='left')

if save_instead_of_plot:
    plt.savefig('figures/PaperFig5_DA_gmt_reconstruction_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()

