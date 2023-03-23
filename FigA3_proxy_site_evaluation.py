#==============================================================================
# Compare the original proxy data vs. proxy reconstructions from the
# reconstruction. This is Fig. A3 in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]

# Select which records to include in this analysis (annual, summerOnly, winterOnly, summer+, winter+)
seasons_to_include = ['annual','summerOnly','winterOnly']


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
recon_mean_proxies      = handle['proxyrecon_mean'].values
prior_mean_proxies      = handle['proxyprior_mean'].values
proxy_values            = handle['proxy_values'].values
proxy_uncertainty       = handle['proxy_uncertainty'].values
proxies_assimilated_all = handle['proxies_assimilated'].values
proxy_metadata          = handle['proxy_metadata'].values
handle.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Select proxies of the desired seasonality
if 'all' not in seasons_to_include:
    #
    # Get indices of the proxy records with the desired seasonality
    proxy_seasonality = proxy_metadata[:,5]
    ind_selected = []
    for i in range(len(proxy_seasonality)):
        if proxy_seasonality[i] in seasons_to_include:
            ind_selected.append(i)
    #
    ind_selected = np.array(ind_selected)
    #
    # Reduce variables to only include records of the selected seasonalities
    recon_mean_proxies      = recon_mean_proxies[:,ind_selected]
    prior_mean_proxies      = prior_mean_proxies[:,ind_selected]
    proxy_values            = proxy_values[:,ind_selected]
    proxy_uncertainty       = proxy_uncertainty[ind_selected]
    proxies_assimilated_all = proxies_assimilated_all[:,ind_selected]


#%%
# Get proxy metadata
model_res = 10
n_ages = proxy_values.shape[0]
n_proxies = proxy_values.shape[1]

# Figure out which proxies were assimilated
ind_proxies_assimilated = np.where(np.mean(proxies_assimilated_all,axis=0) > 0)[0]
ind_proxies_leftout     = np.where(np.mean(proxies_assimilated_all,axis=0) == 0)[0]

print('Number of proxies...')
print('Assimilated:',len(ind_proxies_assimilated))
print('Omitted:    ',len(ind_proxies_leftout))

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

correlations_for_proxies,mean_errors_for_proxies,RMSE_for_proxies,CE_for_proxies = calc_metrics(proxy_values,recon_mean_proxies)
correlations_for_prior,  mean_errors_for_prior,  RMSE_for_prior,  CE_for_prior   = calc_metrics(proxy_values,prior_mean_proxies)

# Calculate the differences
correlations_diff = correlations_for_proxies - correlations_for_prior
mean_errors_diff  = mean_errors_for_proxies  - mean_errors_for_prior
RMSE_diff         = RMSE_for_proxies         - RMSE_for_prior
CE_diff           = CE_for_proxies           - CE_for_prior


#%%
print('Full ranges of values:')
print('  Correlation:',np.nanmin(correlations_for_proxies),np.nanmax(correlations_for_proxies))
print('  CE         :',np.nanmin(CE_for_proxies),          np.nanmax(CE_for_proxies))
print('  RMSE       :',np.nanmin(RMSE_for_proxies),        np.nanmax(RMSE_for_proxies))
print('  Mean error :',np.nanmin(mean_errors_for_proxies), np.nanmax(mean_errors_for_proxies))

print('N values:',sum(np.isfinite(correlations_for_proxies)),sum(np.isfinite(CE_for_proxies)),sum(np.isfinite(RMSE_for_proxies)),sum(np.isfinite(mean_errors_for_proxies)))


#%% FIGURES
plt.style.use('ggplot')

# Make a plot showing the distributions of proxy metrics
corr_all,CE_all,RMSE_all,title_txt = correlations_for_proxies,CE_for_proxies,RMSE_for_proxies,'reconstructions'
def hist_of_metrics(corr_all,CE_all,RMSE_all,title_txt,filename_txt):
    #
    f, ax = plt.subplots(3,1,figsize=(12,12),sharex=False,sharey=False)
    ax = ax.ravel()
    bins_corr      = np.arange(-1,1.01,.05)
    bins_CE        = np.arange(-5,1.1,.1)
    bins_meanerror = np.arange(0,10.1,.25)
    #
    corr_assim = corr_all[ind_proxies_assimilated]
    CE_assim   = CE_all[ind_proxies_assimilated]
    RMSE_assim = RMSE_all[ind_proxies_assimilated]
    #
    corr_leftout = corr_all[ind_proxies_leftout]
    CE_leftout   = CE_all[ind_proxies_leftout]
    RMSE_leftout = RMSE_all[ind_proxies_leftout]
    #
    print('Values plotted (corr, CE, RMSE):')
    print('Assimilated:',sum(np.isfinite(corr_assim)),  sum(np.isfinite(CE_assim)),  sum(np.isfinite(RMSE_assim)))
    print('Omitted:    ',sum(np.isfinite(corr_leftout)),sum(np.isfinite(CE_leftout)),sum(np.isfinite(RMSE_leftout)))
    #
    if title_txt == 'change':
        full_title_txt = 'Change in proxy metrics'
        symbol_txt = '$\Delta$'
        bins_meanerror = np.arange(-1,1.1,.1)
    else:
        full_title_txt = 'Comparison of each proxy and '+title_txt
        symbol_txt = ''
    #
    ax[0].hist(corr_assim,bins=bins_corr,color='tab:red', alpha=0.5,label='Assimilated;  median: '+str('{:.2f}'.format(np.nanmedian(corr_assim)))+',  full range: '+str('{:.2f}'.format(np.nanmin(corr_assim)))+' - '+str('{:.2f}'.format(np.nanmax(corr_assim))),zorder=2)
    if len(ind_proxies_leftout) > 0: ax[0].hist(corr_leftout,bins=bins_corr,color='tab:blue',alpha=0.5,label='Omitted;        median: '+str('{:.2f}'.format(np.nanmedian(corr_leftout)))+',  full range: '+str('{:.2f}'.format(np.nanmin(corr_leftout)))+' - '+str('{:.2f}'.format(np.nanmax(corr_leftout))),zorder=2)
    ax[0].axvline(x=np.nanmedian(corr_assim),color='tab:red', alpha=1,linestyle=':',linewidth=2)
    ax[0].axvline(x=np.nanmedian(corr_leftout),color='tab:blue',alpha=1,linestyle=':',linewidth=2)
    ax[0].axvline(x=0,color='k',alpha=1,linestyle='--',linewidth=1)
    ax[0].set_xlim(np.min(bins_corr),np.max(bins_corr))
    ax[0].set_title('(a) '+symbol_txt+'Correlations',loc='left',fontsize=16)
    ax[0].set_xlabel(symbol_txt+'Correlation',fontsize=16)
    ax[0].legend(loc=2,fontsize=11)
    #
    ax[1].hist(CE_assim,bins=bins_CE,color='tab:red', alpha=0.5,label='Assimilated;  median: '+str('{:.2f}'.format(np.nanmedian(CE_assim)))+',  full range: '+str('{:.2f}'.format(np.nanmin(CE_assim)))+' - '+str('{:.2f}'.format(np.nanmax(CE_assim))),zorder=2)
    if len(ind_proxies_leftout) > 0: ax[1].hist(CE_leftout,bins=bins_CE,color='tab:blue',alpha=0.5,label='Omitted;        median: '+str('{:.2f}'.format(np.nanmedian(CE_leftout)))+',  full range: '+str('{:.2f}'.format(np.nanmin(CE_leftout)))+' - '+str('{:.2f}'.format(np.nanmax(CE_leftout))),zorder=2)
    ax[1].axvline(x=np.nanmedian(CE_assim),color='tab:red', alpha=1,linestyle=':',linewidth=2)
    ax[1].axvline(x=np.nanmedian(CE_leftout),color='tab:blue',alpha=1,linestyle=':',linewidth=2)
    ax[1].axvline(x=0,color='k',alpha=1,linestyle='--',linewidth=1)
    ax[1].set_xlim(np.min(bins_CE),np.max(bins_CE))
    ax[1].set_title('(b) '+symbol_txt+'CE',loc='left',fontsize=16)
    ax[1].set_xlabel(symbol_txt+'CE',fontsize=16)
    ax[1].legend(loc=2,fontsize=11)
    #
    ax[2].hist(RMSE_assim,bins=bins_meanerror,color='tab:red', alpha=0.5,label='Assimilated;  median: '+str('{:.2f}'.format(np.nanmedian(RMSE_assim)))+'$^\circ$C,  full range: '+str('{:.2f}'.format(np.nanmin(RMSE_assim)))+' - '+str('{:.2f}'.format(np.nanmax(RMSE_assim)))+'$^\circ$C',zorder=2)
    if len(ind_proxies_leftout) > 0: ax[2].hist(RMSE_leftout,bins=bins_meanerror,color='tab:blue',alpha=0.5,label='Omitted;        median: '+str('{:.2f}'.format(np.nanmedian(RMSE_leftout)))+'$^\circ$C,  full range: '+str('{:.2f}'.format(np.nanmin(RMSE_leftout)))+' - '+str('{:.2f}'.format(np.nanmax(RMSE_leftout)))+'$^\circ$C',zorder=2)
    ax[2].axvline(x=np.nanmedian(RMSE_assim),color='tab:red', alpha=1,linestyle=':',linewidth=2)
    ax[2].axvline(x=np.nanmedian(RMSE_leftout),color='tab:blue',alpha=1,linestyle=':',linewidth=2)
    ax[2].set_xlim(np.min(bins_meanerror),np.max(bins_meanerror))
    ax[2].set_title('(c) '+symbol_txt+'RMSE',loc='left',fontsize=16)
    ax[2].set_xlabel('$\Delta$T ($^\circ$C)',fontsize=16)
    ax[2].legend(loc=1,fontsize=11)
    #
    for i in range(3):
        ax[i].set_ylabel('Frequency',fontsize=16)
        ax[i].tick_params(axis='both',which='major',labelsize=14)
    #
    plt.suptitle(full_title_txt+', $N_{assim}$='+str(len(ind_proxies_assimilated)),fontsize=20)
    f.tight_layout()
    f.subplots_adjust(top=.9)
    #
    if save_instead_of_plot:
        plt.savefig('figures/'+filename_txt+'_proxy_data_vs_'+title_txt.replace(' ','_')+'_by_proxy_hist_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    #
    #
    # Print the verification values
    if title_txt == 'reconstructed proxy':
        print(' ===== Values for comparison, '+exp_txt+', '+str('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(np.nanmedian(corr_assim),np.nanmedian(corr_leftout),np.nanmedian(CE_assim),np.nanmedian(CE_leftout),np.nanmedian(RMSE_assim),np.nanmedian(RMSE_leftout))))


hist_of_metrics(correlations_for_proxies,CE_for_proxies,RMSE_for_proxies,'reconstructed proxy','PaperFigA3')
hist_of_metrics(correlations_diff,       CE_diff,       RMSE_diff,       'change','extra_FigS7')
hist_of_metrics(correlations_for_prior,  CE_for_prior,  RMSE_for_prior,  'prior','extra_FigS7')
