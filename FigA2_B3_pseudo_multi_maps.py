#==============================================================================
# This makes maps of skill metrics for multiple pseudoproxy experiments. This
# makes Figs. A2 and B3 in the paper. The input files are made in the script
# FigS5_pseudo_gmt_and_map.py.
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

save_instead_of_plot = False

# A function to create multiple correlation map figures
def multi_corr_maps(experiments_txt):
    #
    ### LOAD DATA
    # Set the filenames
    filenames = {}
    if experiments_txt == 'prior_and_grid':
        filenames['Exp. 1'] = 'output_holocene_recon_2022-04-04_16:55:44.482606_pseudo_default.nc'
        filenames['Exp. 2'] = 'output_holocene_recon_2022-04-04_17:24:25.021686_pseudo_basicgrid.nc'
        filenames['Exp. 3'] = 'output_holocene_recon_2022-04-04_16:57:38.485355_pseudo_goodphysics.nc'
        filenames['Exp. 4'] = 'output_holocene_recon_2022-04-04_17:24:11.800109_pseudo_basicgrid_goodphysics.nc'
        filename_txt = 'PaperFigA2'
    elif experiments_txt == 'timeconstant_and_locrad':
        filenames['Exp. 5'] = 'output_holocene_recon_2022-04-04_17:04:42.441901_pseudo_prior_constant_mean.nc'
        filenames['Exp. 6'] = 'output_holocene_recon_2022-04-04_17:19:45.659888_pseudo_prior_constant.nc'
        filenames['Exp. 7'] = 'output_holocene_recon_2022-04-05_01:40:10.392185_pseudo_locrad_25k.nc'
        filenames['Exp. 8'] = 'output_holocene_recon_2022-04-04_17:09:47.364905_pseudo_prior_3010.nc'
        filename_txt = 'PaperFigB3'
    elif experiments_txt == 'window_length':
        filenames['1010yr'] = 'output_holocene_recon_2022-04-04_17:13:09.999569_pseudo_prior_1010.nc'
        filenames['3010yr'] = 'output_holocene_recon_2022-04-04_17:09:47.364905_pseudo_prior_3010.nc'
        filenames['7010yr'] = 'output_holocene_recon_2022-04-04_17:09:51.890160_pseudo_prior_7010.nc'
        filenames['9010yr'] = 'output_holocene_recon_2022-04-04_17:15:51.842729_pseudo_prior_9010.nc'
        filename_txt = 'Extra_fig'
    #
    exp_names = list(filenames.keys())
    #
    # Load the Holocene reconstruction
    data_dir = '/home/mpe32/analysis/15_Holocene_Reconstruction/analyze_results/paper_figures/data/'
    data = {}
    for exp in exp_names:
        data[exp] = {}
        handle = xr.open_dataset(data_dir+filenames[exp],decode_times=False)
        data[exp]['R']          = handle['tas_corr'].values
        data[exp]['Mean_error'] = handle['tas_mean_error'].values
        data[exp]['RMSE']       = handle['tas_rmse'].values
        data[exp]['CE']         = handle['tas_ce'].values
        data[exp]['proxy_lats'] = handle['proxy_lats'].values
        data[exp]['proxy_lons'] = handle['proxy_lons'].values
        data[exp]['lat']        = handle['lat'].values
        data[exp]['lon']        = handle['lon'].values
        handle.close()
        print('Number of proxies for '+exp+': '+str(len(data[exp]['proxy_lats'])))
    #
    #
    ### FIGURES
    plt.style.use('ggplot')
    letters = ['a','b','c','d','e','f','g','h']
    #
    values_to_plot = ['R','CE']
    #values_to_plot = ['R','RMSE']
    #
    # Make maps of metrics in space
    plt.figure(figsize=(17,22))
    ax = {}
    counter = 0
    for i,exp in enumerate(exp_names):
        for j,value_name in enumerate(values_to_plot):
            #
            if   value_name == 'R':    bound_min = -1; bound_max = 1; cmap = 'Spectral_r'; extend = 'neither'
            elif value_name == 'CE':   bound_min = -1; bound_max = 1; cmap = 'Spectral_r'; extend = 'both'
            elif value_name == 'RMSE': bound_min = 0;  bound_max = 1; cmap = 'viridis';    extend = 'max'
            #
            # Compute a global mean
            metric_global = da_utils.global_mean(data[exp][value_name],data[exp]['lat'],0,1)
            #
            # Make a map of correlations at every point
            ax[counter] = plt.subplot2grid((4,2),(i,j),projection=ccrs.Robinson(central_longitude=0))
            metric_cyclic,lon_cyclic = cutil.add_cyclic_point(data[exp][value_name],coord=data[exp]['lon'])
            map1 = ax[counter].contourf(lon_cyclic,data[exp]['lat'],metric_cyclic,cmap=cmap,levels=np.arange(bound_min,bound_max+.1,.1),extend=extend,transform=ccrs.PlateCarree())
            ax[counter].scatter(data[exp]['proxy_lons'],data[exp]['proxy_lats'],5,c='k',marker='o',alpha=1,transform=ccrs.PlateCarree())
            if counter >= 6:
                colorbar = plt.colorbar(map1,orientation='horizontal',ax=ax[counter],fraction=0.08,pad=0.02)
                colorbar.set_label(value_name,fontsize=18)
                colorbar.ax.tick_params(labelsize=18)
                colorbar.ax.set_facecolor('none')
            ax[counter].coastlines()
            ax[counter].gridlines(color='k',linestyle='--',draw_labels=False)
            ax[counter].set_title('('+letters[counter]+') '+exp+', '+value_name+' (mean='+str('{:.2f}'.format(metric_global))+')',fontsize=22)
            counter += 1
    #
    plt.tight_layout()
    plt.suptitle('Correlation and '+values_to_plot[1]+' in four pseudoproxy experiments',fontsize=25)
    plt.subplots_adjust(top=.96)
    #
    if save_instead_of_plot:
        plt.savefig('figures/'+filename_txt+'_map_gridded_all_'+experiments_txt+'_'+values_to_plot[0]+'_'+values_to_plot[1]+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    

#%% Run the function
multi_corr_maps('prior_and_grid')
multi_corr_maps('timeconstant_and_locrad')
multi_corr_maps('window_length')

