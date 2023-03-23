#==============================================================================
# Make a histogram of median proxy resolution. This is Fig. 1 in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import lipd

save_instead_of_plot = False


#%% LOAD DATA

# Load the Temp12k proxy metadata
dir_proxies_temp12k = '/projects/pd_lab/data/data_assimilation/proxies/temp12k/'
file_to_open = open(dir_proxies_temp12k+'Temp12k1_0_2.pkl','rb')
proxies_all_12k = pickle.load(file_to_open)['D']
file_to_open.close()

# Extract the time series and use only those which are in Temp12k and in units of degC
all_ts_12k = lipd.extractTs(proxies_all_12k)

# Fix the GISP2 ages - Note: this is a temporary fix, since lipd isn't loading the right ages.
for i in range(len(all_ts_12k)):
    if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'age'): gisp2_ages = all_ts_12k[i]['paleoData_values']

for i in range(len(all_ts_12k)):
    if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'temperature') and (np.max(np.array(all_ts_12k[i]['age']).astype(float)) < 50):
        print('Fixing GISP2 ages:',all_ts_12k[i]['paleoData_variableName'],', Index:',i)
        all_ts_12k[i]['age'] = gisp2_ages

filtered_ts = lipd.filterTs(all_ts_12k, 'paleoData_inCompilation == Temp12k')
filtered_ts = lipd.filterTs(filtered_ts,'paleoData_units == degC')


#%% CALCULATIONS

# Load every file and find the temporal resolution of the proxies
n_proxies = len(filtered_ts)
resolution_median_all   = np.zeros((n_proxies)); resolution_median_all[:]   = np.nan
resolution_min_all      = np.zeros((n_proxies)); resolution_min_all[:]      = np.nan
resolution_metadata_all = np.zeros((n_proxies)); resolution_metadata_all[:] = np.nan
archivetype_all = []
proxytype_all   = []
counter = 0
for i in range(n_proxies):
    #
    # Get proxy data
    proxy_ages   = np.array(filtered_ts[i]['age']).astype(float)
    proxy_values = np.array(filtered_ts[i]['paleoData_values']).astype(float)
    resolution_metadata = filtered_ts[i]['paleoData_hasResolution_hasMedianValue']
    archivetype_all.append(filtered_ts[i]['archiveType'])
    try:    proxytype_all.append(filtered_ts[i]['paleoData_proxy'])
    except: proxytype_all.append('Not given')
    #
    # If any NaNs exist in the ages or values, remove those values
    proxy_ages = proxy_ages[np.isfinite(proxy_ages) & np.isfinite(proxy_values)]
    proxy_ages = np.array([*set(proxy_ages)])
    #
    # Sort the data so that ages go from newest to oldest
    ind_sorted = np.argsort(proxy_ages)
    proxy_ages = proxy_ages[ind_sorted]
    proxy_ages = proxy_ages[(proxy_ages >= 0) & (proxy_ages <= 12000)]
    #
    # Find the difference in ages
    age_diff = proxy_ages[1:]-proxy_ages[:-1]
    #
    # Save to common variables
    resolution_median_all[counter]   = np.nanmedian(age_diff)
    resolution_min_all[counter]      = np.nanmin(age_diff)
    resolution_metadata_all[counter] = resolution_metadata
    counter += 1

# Take a look at the resolutions shorter than decades and longer than millenia
indices_lt_10yr   = (resolution_median_all <= 10)
indices_gt_200yr  = (resolution_median_all >= 200)
indices_gt_500yr  = (resolution_median_all >= 500)
indices_gt_1000yr = (resolution_median_all >= 1000)

print('Median of medians:',np.median(resolution_median_all))
print('---')
print('Total proxy records:',n_proxies)
print('NaN values (n='+str(sum(np.isnan(resolution_median_all)))+')')
print('Min resolutions <10 year (n='+str(sum(resolution_min_all < 10))+')')
print('Median resolutions <= 10 year (n='+str(sum(indices_lt_10yr))+'):',np.sort(resolution_median_all[indices_lt_10yr]))
print('Median resolutions >= 200 year (n='+str(sum(indices_gt_200yr))+'):',np.sort(resolution_median_all[indices_gt_200yr]))
print('Median resolutions >= 500 year (n='+str(sum(indices_gt_500yr))+'):',np.sort(resolution_median_all[indices_gt_500yr]))
print('Median resolutions >= 1000 year (n='+str(sum(indices_gt_1000yr))+'):',np.sort(resolution_median_all[indices_gt_1000yr]))

# Count the number of each name
archivetypes,archivecounts = np.unique(archivetype_all,return_counts=True)
count_sort_ind = np.argsort(-archivecounts)
archivetypes_sorted  = archivetypes[count_sort_ind]
archivecounts_sorted = archivecounts[count_sort_ind]

# Plot make lists of median resolutions for each archive type
list_res = []
for type_selected in archivetypes_sorted:
    ind_selected = np.where(np.array(archivetype_all) == type_selected)[0]
    resolution_median_selected = resolution_median_all[ind_selected]
    list_res.append(resolution_median_selected)

# Plot the difference between the calculated median and the metadata median
diff = resolution_metadata_all - resolution_median_all
print(min(diff),max(diff))
plt.plot(diff)


#%% FIGURES
print(min(resolution_median_all),max(resolution_median_all))
plt.style.use('ggplot')

# Plot histograms of values
f,ax1 = plt.subplots(1,1,figsize=(10,5))
bins_values = np.arange(0,801,20)

ax1.hist(resolution_median_all,bins=bins_values,label='Resolution',zorder=2)
ax1.set_xlabel('Median temporal resolution (years)',fontsize=16)
ax1.set_ylabel('# proxies',fontsize=16)
#ax1.set_xlim(xmin=0)
ax1.set_xlim(0,800)
ax1.axvline(x=np.nanmedian(resolution_median_all),color='k',alpha=1,linestyle=':',linewidth=2)

plt.suptitle('Median temporal resolutions of Temp12k proxies\nMedian: '+str('{:.2f}'.format(np.median(resolution_median_all)))+' years, Range: '+str('{:.2f}'.format(min(resolution_median_all)))+' - '+str('{:.2f}'.format(max(resolution_median_all)))+' years',fontsize=18)
f.tight_layout()
f.subplots_adjust(top=.85)
if save_instead_of_plot:
    plt.savefig('figures/resolution_histogram.png',dpi=200,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%%
# Add a color to the default color cycle
default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=default_colors+['k'])

# Plot histograms of values
f,ax1 = plt.subplots(1,1,figsize=(10,5))
bins_values = np.arange(0,801,20)

ax1.hist(list_res,bins=bins_values,stacked=True)  # v1
#for i,archive_label in enumerate(archivetypes_sorted): ax1.hist(list_res[i],bins=bins_values)  # v2
ax1.legend(archivetypes_sorted)
ax1.set_xlabel('Median temporal resolution (years)',fontsize=16)
ax1.set_ylabel('# proxies',fontsize=16)
ax1.set_xlim(0,800)

plt.suptitle('Median temporal resolutions of Temperature 12k proxies during 0-12 ka',fontsize=16)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot:
    plt.savefig('figures/PaperFig1_resolution_by_type_hist.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()
