#=============================================================================
# This script compares the prior and posterier in three different data
# assimilation experiments. It makes Fig. B1 for the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#=============================================================================

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

save_instead_of_plot = False


#%% LOAD DATA

# Pick the reconstructions to analyze
filenames = {}
filenames['time-varying prior']                  = 'holocene_reconstruction.nc'
filenames['time-varying prior w/ constant mean'] = 'holocene_recon_2022-03-30_09:51:26.629557_prior_constant_mean.nc'
filenames['time-constant prior']                 = 'holocene_recon_2022-03-30_10:19:55.927967_prior_constant.nc'

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
experiments = list(filenames.keys())

handle = xr.open_dataset(recon_dir+'results/'+filenames[experiments[0]],decode_times=False)
lat     = handle['lat'].values
lon     = handle['lon'].values
ages_da = handle['ages'].values
handle.close()

recon_gmt_orig,prior_gmt_orig = {},{}
for exp in experiments:
    handle = xr.open_dataset(recon_dir+'results/'+filenames[exp],decode_times=False)
    recon_gmt_orig[exp] = handle['recon_tas_global_mean'].values
    prior_gmt_orig[exp] = handle['prior_tas_global_mean'].values
    handle.close()


#%% CALCULATIONS

# Remove the mean of the 50-150 yr BP age interval from the reconstruction
indices_ref_recon = np.where((ages_da >= 50) & (ages_da <= 150))[0]
recon_gmt,prior_gmt = {},{}
for exp in experiments:
    recon_gmt[exp] = recon_gmt_orig[exp] - np.mean(recon_gmt_orig[exp][indices_ref_recon,:])
    prior_gmt[exp] = prior_gmt_orig[exp] - np.mean(prior_gmt_orig[exp][indices_ref_recon,:])


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
    calc_6ka_anom(np.mean(recon_gmt[exp],axis=1),ages_da,exp)


#%% FIGURES
plt.style.use('ggplot')
colors = ['tab:red','tab:blue','tab:green']

# Plot time series of the reconstructions
f, ax = plt.subplots(3,1,figsize=(12,15),sharex=True)
ax = ax.ravel()

letters = ['a','b','c']

for i,exp in enumerate(experiments):
    range_prior_all = ax[i].fill_between(ages_da,np.nanmin(prior_gmt_orig[exp],axis=1),          np.nanmax(prior_gmt_orig[exp],axis=1),          color='gray',alpha=0.25)
    range_prior     = ax[i].fill_between(ages_da,np.nanpercentile(prior_gmt_orig[exp],16,axis=1),np.nanpercentile(prior_gmt_orig[exp],84,axis=1),color='gray',alpha=0.25)
    line_prior,     = ax[i].plot(ages_da,np.nanmean(prior_gmt_orig[exp],axis=1),color='gray',    linewidth=3,label='Prior')
    range_da_all    = ax[i].fill_between(ages_da,np.nanmin(recon_gmt_orig[exp],axis=1),          np.nanmax(recon_gmt_orig[exp],axis=1),          color='tab:blue',alpha=0.5)
    range_da        = ax[i].fill_between(ages_da,np.nanpercentile(recon_gmt_orig[exp],16,axis=1),np.nanpercentile(recon_gmt_orig[exp],84,axis=1),color='tab:blue',alpha=0.5)
    line_da,        = ax[i].plot(ages_da,np.nanmean(recon_gmt_orig[exp],axis=1),color='tab:blue',linewidth=3,label='Posterior')
    if i != 0: line_da_ref, = ax[i].plot(ages_da,np.nanmean(recon_gmt_orig['time-varying prior'],axis=1),color='k',linewidth=1)
    if i == 0: ax[i].legend(ncol=2,fontsize=18,loc='lower right')
    ax[i].set_ylim(-3.75,1)
    ax[i].axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
    ax[i].set_ylabel('$\Delta$T ($^\circ$C)',fontsize=20)
    ax[i].tick_params(axis='both',which='major',labelsize=20)
    ax[i].set_title('('+letters[i]+') Using '+exp,loc='left',fontsize=18)

ax[2].set_xlabel('Age (yr BP)',fontsize=20)
ax[2].set_xlim(12000,0)

plt.suptitle('Global-mean temperature anomalies ($^\circ$C),\nin prior and posterior',fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=.9)

if save_instead_of_plot:
    plt.savefig('figures/PaperFigB1_DA_gmt_and_prior_3ways.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Plot time series of the reconstructions
for i,exp in enumerate(experiments):
    if exp == 'time-varying prior': continue
    #
    f,ax1 = plt.subplots(1,1,figsize=(12,12))
    ax1.fill_between(ages_da,np.nanmin(recon_gmt_orig['time-varying prior'],axis=1),np.nanmax(recon_gmt_orig['time-varying prior'],axis=1),color='gray',alpha=0.5)
    ax1.plot(ages_da,np.nanmean(recon_gmt_orig['time-varying prior'],axis=1),color='k',linewidth=3,label='time-varying prior')
    ax1.fill_between(ages_da,np.nanmin(recon_gmt_orig[exp],axis=1),np.nanmax(recon_gmt_orig[exp],axis=1),color='tab:blue',alpha=0.5)
    ax1.plot(ages_da,np.nanmean(recon_gmt_orig[exp],axis=1),color='tab:blue',linewidth=3,label=exp)
    ax1.legend(ncol=1,fontsize=18,loc='lower right')
    ax1.set_xlim(12000,0)
    ax1.set_ylim(-2.2,0.5)
    ax1.axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
    ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=20)
    ax1.tick_params(axis='both',which='major',labelsize=20)
    ax1.set_title(exp+' vs. default',loc='left',fontsize=18)
    ax1.set_xlabel('Age (yr BP)',fontsize=20)
    #
    if save_instead_of_plot:
        plt.savefig('figures/DA_gmt_'+str(i+1)+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()


