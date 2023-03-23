#=============================================================================
# This script compares the Holocene reconstruction to other Holocene
# reconstructions. This is Fig. 13 in the paper.
#    author: Michael P. Erb
#    date  : 3/23/2023
#=============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy import stats

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_reconstruction.nc' # The default experiment
#recon_filename = sys.argv[1]


#%% LOAD DATA

# Load the Holocene reconstruction
recon_dir = '/projects/pd_lab/data/data_assimilation/'
handle = xr.open_dataset(recon_dir+'results/'+recon_filename,decode_times=False)
gmt_da  = handle['recon_tas_global_mean'].values
lat     = handle['lat'].values
lon     = handle['lon'].values
ages_da = handle['ages'].values
handle.close()

# Load 12k data
data_dir = '/home/mpe32/analysis/14_Holocene_proxies/GMST_paper/data/'
handle = xr.open_dataset(data_dir+'final_data/temp12k_alldata.nc',decode_times=False)
ages_kaufman    = handle['age'].values
gmt_kaufman_scc = handle['scc_globalmean'].values
gmt_kaufman_dcc = handle['dcc_globalmean'].values
gmt_kaufman_cps = handle['cps_globalmean'].values
gmt_kaufman_pai = handle['pai_globalmean'].values
gmt_kaufman_gam = handle['gam_globalmean'].values
handle.close()

# Combine all 12k methods into one array
gmt_kaufman = np.concatenate((gmt_kaufman_scc,gmt_kaufman_dcc,gmt_kaufman_cps,gmt_kaufman_pai,gmt_kaufman_gam),axis=1)

# Load other Holocene reconstructions
data_dir_recons = '/projects/pd_lab/data/paleoclimate_reconstructions/Holocene_reconstructions/'

# Load the reconstruction from Shakun et al., 2012
data_shakun = pd.ExcelFile(data_dir_recons+'Shakun_etal_2012/41586_2012_BFnature10915_MOESM60_ESM.xls').parse('TEMPERATURE STACKS').values
ages_shakun     = (data_shakun[:,0]*1000).astype(float)
gmt_shakun      = data_shakun[:,1].astype(float)
onesigma_shakun = data_shakun[:,2].astype(float)

# Load the reconstruction from Marcott et al., 2013
data_marcott = pd.read_excel(data_dir_recons+'Marcott_etal_2013/Marcott.SM.database.S1.xlsx',sheet_name='TEMPERATURE STACKS',engine='openpyxl').values
ages_marcott     = data_marcott[5:,2].astype(float)
gmt_marcott      = data_marcott[5:,3].astype(float)
onesigma_marcott = data_marcott[5:,4].astype(float)

# Load the reconstruction from Bova et al., 2021
data_bova = pd.read_excel(data_dir_recons+'Bova_etal_2021/41586_2020_3155_MOESM4_ESM.xlsx',sheet_name='Sheet1',engine='openpyxl').values
ages_bova         = data_bova[2:14,13].astype(float)*1000
sstsn_bova        = data_bova[2:14,14].astype(float)
sstsn_onestd_bova = data_bova[2:14,15].astype(float)
masst_bova        = data_bova[2:14,16].astype(float)
masst_onestd_bova = data_bova[2:14,17].astype(float)

# Load the reconstruction from Osman  et al., 2021
handle_osman = xr.open_dataset(data_dir_recons+'Osman_etal_2021/LGMR_GMST_ens.nc',decode_times=False)
gmt_osman  = handle_osman['gmst'].values
ages_osman = handle_osman['age'].values
handle_osman.close()


#%% CALCULATIONS

# Get some text from the experiment filename
recon_filename_split = recon_filename.split('.')
exp_txt = recon_filename_split[-2][7:]

# Look at the ages of each reconstruction
print('DA:     ',ages_da[0:5],     ages_da[-5:])
print('Kaufman:',ages_kaufman[0:5],ages_kaufman[-5:])
print('Shakun: ',ages_shakun[0:5], ages_shakun[-5:])
print('Marcott:',ages_marcott[0:5],ages_marcott[-5:])
print('Bova:   ',ages_bova[0:5],   ages_bova[-5:])
print('Osman:  ',ages_osman[0:5],  ages_osman[-5:])


#%%
# Compute the relative mid-Holocene temperature for each reconstruction
def calc_6ka_anom(data_ts,ages_ts,name_txt):
    #
    # Calculate 6-0 ka anomaly
    ind_anom = np.where((ages_ts >= 5500) & (ages_ts <= 6500))[0]
    ind_ref  = np.where((ages_ts >= 0)    & (ages_ts <= 1000))[0]
    anom_6ka = np.mean(data_ts[ind_anom]) - np.mean(data_ts[ind_ref])
    print('6ka anom: ',name_txt,anom_6ka,'; Lengths of intervals:',len(ind_anom),len(ind_ref))
    #
    # Calculate 6-0 ka trend
    ind_0_6ka = np.where((ages_ts >= 0) & (ages_ts <= 6000))[0]
    if len(ind_0_6ka) > 0:
        trend_6ka,intercept,rvalue,pvalue,_ = stats.linregress(ages_ts[ind_0_6ka],data_ts[ind_0_6ka])
        trend_6ka = -1000*trend_6ka
    else:
        trend_6ka = np.nan
    #
    print('6ka trend: ',name_txt,trend_6ka)
    #
    return trend_6ka,anom_6ka

trend_6ka_da,     anom_6ka_da      = calc_6ka_anom(np.mean(gmt_da,axis=1),       ages_da,     'DA')
trend_6ka_kaufman,anom_6ka_kaufman = calc_6ka_anom(np.median(gmt_kaufman,axis=1),ages_kaufman,'Kaufman')
trend_6ka_shakun, anom_6ka_shakun  = calc_6ka_anom(gmt_shakun,                   ages_shakun, 'Shakun')
trend_6ka_marcott,anom_6ka_marcott = calc_6ka_anom(gmt_marcott,                  ages_marcott,'Marcott')
trend_6ka_bova,   anom_6ka_bova    = calc_6ka_anom(masst_bova,                   ages_bova,   'Bova')
trend_6ka_osman,  anom_6ka_osman   = calc_6ka_anom(np.mean(gmt_osman,axis=0),    ages_osman,  'Osman')


#%%

# Remove the mean of the 50-150 yr BP age interval from the reconstructions
ref_ages = [50,150]
indices_ref_recon   = np.where((ages_da      >= ref_ages[0]) & (ages_da      <= ref_ages[1]))[0]
indices_ref_marcott = np.where((ages_marcott >= ref_ages[0]) & (ages_marcott <= ref_ages[1]))[0]
indices_ref_osman   = np.where((ages_osman   >= ref_ages[0]) & (ages_osman   <= ref_ages[1]))[0]

gmt_da      = gmt_da      - np.mean(gmt_da[indices_ref_recon,:])
gmt_marcott = gmt_marcott - np.mean(gmt_marcott[indices_ref_marcott])
gmt_osman   = gmt_osman   - np.mean(np.mean(gmt_osman[:,indices_ref_osman],axis=1),axis=0)

# Find the difference in means between two time series during their period of overlap
def mean_of_overlap(ts1,ages1,ts2,ages2):
    overlap_age_min = np.max([np.min(ages1),np.min(ages2)])
    overlap_age_max = np.min([np.max(ages1),np.max(ages2)])
    ts1_mean_overlap_period = np.mean(ts1[np.where((ages1 >= overlap_age_min) & (ages1 <= overlap_age_max))[0]])
    ts2_mean_overlap_period = np.mean(ts2[np.where((ages2 >= overlap_age_min) & (ages2 <= overlap_age_max))[0]])
    difference_in_means = ts1_mean_overlap_period - ts2_mean_overlap_period
    return difference_in_means

# The Shukun reconstruction stops at 6.5 ka.  Align it to the Marcott reconstruction using the period of overlap
difference_in_means_shakun = mean_of_overlap(gmt_shakun,ages_shakun,gmt_marcott,ages_marcott)
gmt_shakun = gmt_shakun - difference_in_means_shakun


#%% FIGURES
plt.style.use('ggplot')

# Plot time series of the reconstructions
f, ax1 = plt.subplots(1,1,figsize=(16,12),sharex=True)

range_shakun  = ax1.fill_between(ages_shakun, gmt_shakun-onesigma_shakun,             gmt_shakun+onesigma_shakun,             color='tab:purple', alpha=0.5)
range_marcott = ax1.fill_between(ages_marcott,gmt_marcott-onesigma_marcott,           gmt_marcott+onesigma_marcott,           color='deepskyblue',alpha=0.5)
range_kaufman = ax1.fill_between(ages_kaufman,np.nanpercentile(gmt_kaufman,16,axis=1),np.nanpercentile(gmt_kaufman,84,axis=1),color='darkgreen',  alpha=0.5)
range_bova    = ax1.fill_between(ages_bova,   masst_bova-masst_onestd_bova,           masst_bova+masst_onestd_bova,           color='tab:olive',  alpha=0.5)
range_osman   = ax1.fill_between(ages_osman,  np.nanpercentile(gmt_osman,16,axis=0),  np.nanpercentile(gmt_osman,84,axis=0),  color='tab:red',    alpha=0.5)
range_da      = ax1.fill_between(ages_da,     np.nanpercentile(gmt_da,2.5,axis=1),    np.nanpercentile(gmt_da,97.5,axis=1),   color='k',          alpha=0.25)
range_da      = ax1.fill_between(ages_da,     np.nanpercentile(gmt_da,16,axis=1),     np.nanpercentile(gmt_da,84,axis=1),     color='k',          alpha=0.5)

line_shakun,  = ax1.plot(ages_shakun, gmt_shakun,                      color='tab:purple', linewidth=3,label='Shakun et al. 2012   (6 - 0.5 ka = NA)')
line_marcott, = ax1.plot(ages_marcott,gmt_marcott,                     color='deepskyblue',linewidth=3,label='Marcott et al. 2013   (6 - 0.5 ka = '+str('{:.2f}'.format(anom_6ka_marcott))+'$^\circ$C)')
line_kaufman, = ax1.plot(ages_kaufman,np.nanmedian(gmt_kaufman,axis=1),color='darkgreen',  linewidth=3,label='Kaufman et al. 2020 (6 - 0.5 ka = '+str('{:.2f}'.format(anom_6ka_kaufman))+'$^\circ$C)')
line_bova,    = ax1.plot(ages_bova,   masst_bova,                      color='tab:olive',  linewidth=3,label='Bova et al. 2021       (6 - 0.5 ka = '+str('{:.2f}'.format(anom_6ka_bova))+'$^\circ$C)')
line_osman,   = ax1.plot(ages_osman,  np.nanmean(gmt_osman,axis=0),    color='tab:red',    linewidth=3,label='Osman et al. 2021    (6 - 0.5 ka = '+str('{:.2f}'.format(anom_6ka_osman))+'$^\circ$C)')
line_da,      = ax1.plot(ages_da,     np.nanmean(gmt_da,axis=1),       color='k',          linewidth=3,label='Holocene DA             (6 - 0.5 ka = '+str('{:.2f}'.format(anom_6ka_da))+'$^\circ$C)')

ax1.axhline(y=0,color='k',linewidth=1,linestyle='dashed',alpha=0.5)
ax1.legend(loc='lower right',ncol=1,fontsize=16)
ax1.set_xlabel('Age (yr BP)',fontsize=20)
ax1.set_ylabel('$\Delta$ Temperature ($^\circ$C)',fontsize=20)
ax1.set_xlim(ages_da[-1],-60)
ax1.set_ylim(-4,1.25)
ax1.set_title('Temperature composites',fontsize=26)
ax1.tick_params(axis='both',which='major',labelsize=20)

if save_instead_of_plot:
    plt.savefig('figures/PaperFig13_DA_gmt_comparison_'+exp_txt+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()

