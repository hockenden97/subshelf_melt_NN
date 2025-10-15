import glob
import os
import numpy as np
import xarray as xr
import time
import sys 

nemo_run = sys.argv[1]
year_min = sys.argv[2]
year_max = sys.argv[3]
print(nemo_run, year_min, year_max)
# Options are 'OPM016', 'OPM018', 'OPM021', 'ctrl94', 'isf94', 'isfru94'

# Set all the filepaths that will be required 
filepath_base = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/'
filepath_nn_input = filepath_base + "data/processing_ho/nn_input_"

# Set all the filepaths that will be required 
if nemo_run == 'OPM016':
    filepath_nemo = '/bettik/burgardc/DATA/BASAL_MELT_PARAM/raw/NEMO_eORCA025.L121-OPM016/'
    file_extension = '*gridT*.nc'
    filepath_mesh_mask = filepath_nemo + 'eORCA025.L121-' + nemo_run + '_mesh_mask.nc'
    monthly = False
elif nemo_run == 'OPM018':
    filepath_nemo = '/bettik/burgardc/DATA/BASAL_MELT_PARAM/raw/NEMO_eORCA025.L121-OPM018/'
    file_extension = '*gridT*.nc'
    filepath_mesh_mask = filepath_nemo + 'eORCA025.L121-' + nemo_run + '_mesh_mask.nc'
    monthly = False
elif nemo_run == 'OPM021':
    filepath_nemo = '/bettik/burgardc/DATA/BASAL_MELT_PARAM/raw/NEMO_eORCA025.L121-OPM021/'
    file_extension = '*gridT*.nc'
    filepath_mesh_mask = filepath_nemo + 'eORCA025.L121-' + nemo_run + '_mesh_mask.nc'
    monthly = False
elif nemo_run == 'ctrl94':
    filepath_nemo = '/bettik/burgardc/DATA/SUMMER_PAPER/raw/CHRISTOPH_DATA/'
    file_extension = 'fwfisf*' + nemo_run + '*.nc'
    filepath_mesh_mask = filepath_nemo + 'mesh_mask.nc'
    filepath_cfg = filepath_nemo + 'domain_cfg_eANT025.L121.nc'
    monthly = False
elif nemo_run == 'isf94':
    filepath_nemo = '/bettik/burgardc/DATA/SUMMER_PAPER/raw/CHRISTOPH_DATA/'
    file_extension = 'fwfisf*' + nemo_run + '*.nc'
    filepath_mesh_mask = filepath_nemo + 'mesh_mask.nc'
    filepath_cfg = filepath_nemo + 'domain_cfg_eANT025.L121.nc'
    monthly = False
elif nemo_run == 'isfru94':
    filepath_nemo = '/bettik/burgardc/DATA/SUMMER_PAPER/raw/CHRISTOPH_DATA/'
    file_extension = 'fwfisf*' + nemo_run + '*.nc'
    filepath_mesh_mask = filepath_nemo + 'mesh_mask.nc'
    filepath_cfg = filepath_nemo + 'domain_cfg_eANT025.L121.nc'
    monthly = False
else:
    filepath_pierre = []
    file_extension = []
    print('Help, I do not know this nemo_run')
print(nemo_run, filepath_nemo)

filepath_mask = filepath_base +  'AIAI_data/Masks/'+nemo_run+'_'+'geometric_masks.nc'
# A geometry file with the required variables for the NN 
filepath_geomvars = filepath_base + 'AIAI_data/Masks/'+nemo_run+'_'+'geom_vars.nc'
print(filepath_mask)
print(filepath_geomvars)

def load_geom_files(filepath_geomvars, filepath_mask, join_ice_shelves = False):
    geoms = xr.open_dataset(filepath_geomvars)
    masks = xr.open_dataset(filepath_mask)
    print('You have loaded:', filepath_geomvars)
    print('You have loaded:', filepath_mask)
    bN_00 = masks.basins_NEMO
    # Some ice shelves which should potentially be joined together into one bigger ice shelf
    if join_ice_shelves == True:
        # Dotson and Crosson?
        #basins_NEMO[basins_NEMO == 101] = 129
        # Abbot Ice Shelf
        bN_01 = xr.where(bN_00 == 109, 143, bN_00)
        # George VI
        bN_02 = xr.where(bN_01 == 112, 125, bN_01)
        # Lambert 
        basins_merged = xr.where(bN_02 == 20, 103, bN_02)
        print('Ice shelves joined, as requested')
    else:
        basins_merged = bN_00    
    return masks, basins_merged, geoms

def load_TSmelt_chris(filepath_nemo, nemo_run, year):
    if nemo_run == 'ctrl94':
        fp_end = '-1y_1982-2013.nc'
    elif nemo_run in ('isf94', 'isfru94'):
        fp_end = '-1y_2014-2100.nc'
    filepath_so = filepath_nemo + 'so-eANT025.L121-' + nemo_run + fp_end
    print('You have loaded:', filepath_so)
    filepath_thetao = filepath_nemo + 'thetao-eANT025.L121-' + nemo_run + fp_end
    print('You have loaded:', filepath_thetao)
    filepath_fwfisf = filepath_nemo + 'fwfisf-eANT025.L121-' + nemo_run + fp_end
    print('You have loaded:', filepath_fwfisf)
    so_all = xr.open_dataset(filepath_so)
    so_yr = so_all.where(so_all.time_counter.dt.year == year_of_interest, drop=True)
    so_int = so_yr.squeeze('time_counter').drop('time_counter').so
    so_int = so_int.rename({'nav_lon_grid_T': 'nav_lon', 'nav_lat_grid_T': 'nav_lat'})
    thetao_all = xr.open_dataset(filepath_thetao)
    thetao_yr = thetao_all.where(thetao_all.time_counter.dt.year == year_of_interest, drop=True)
    thetao_int = thetao_yr.squeeze('time_counter').drop('time_counter').thetao
    thetao_int = thetao_int.rename({'nav_lon_grid_T': 'nav_lon', 'nav_lat_grid_T': 'nav_lat'})
    fwfisf_all = xr.open_dataset(filepath_fwfisf)
    fwfisf_yr = fwfisf_all.where(fwfisf_all.time_counter.dt.year == year_of_interest, drop=True)
    fwfisf = fwfisf_yr.squeeze('time_counter').drop('time_counter').fwfisf
    fwfisf = fwfisf.rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    # Replace 0 values with nan values as in Pierre's simulations
    thetao = thetao_int.where(so_int != 0, np.nan)
    so = so_int.where(so_int != 0, np.nan)
    print(year, 'extracted')
    return so, thetao, fwfisf

def load_TSmelt_normal(filepath_nemo, nemo_run, year):
    filepath_gridT_varofint = filepath_nemo + 'eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                    '.1y_gridT_varofint.nc'
    filepath_flxT_varofint = filepath_nemo + 'eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                    '.1y_flxT_varofint.nc'
    # Load in the simulation files 
    gridT_varofint = xr.open_dataset(filepath_gridT_varofint)
    print('You have loaded:', filepath_gridT_varofint)
    flxT_varofint = xr.open_dataset(filepath_flxT_varofint)
    print('You have loaded:', filepath_flxT_varofint)
    # And cut out variables which are not needed
    gridT_varofint = gridT_varofint.squeeze('time_counter').sel(bnds=1).drop('time_counter')
    flxT_varofint = flxT_varofint.squeeze('time_counter').sel(bnds=1).drop('time_counter')
    # Crop out most of the world and just save Antarctica 
    # Specify the coordinates needed
    min_lat = -52.2
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    print('Cropped to Antarctica only')
    # Take this slice 
    gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))[['votemper','vosaline']]
    flxT2 = flxT_varofint.sel(y = slice(0,y_trim+1))[['sowflisf_cav']]
    gridT2 = gridT2.sel(x = slice(1,1441))
    flxT2 = flxT2.sel(x = slice(1,1441))
    # Close the original dataset
    gridT_varofint.close()
    so = gridT2.vosaline
    thetao = gridT2.votemper
    fwfisf = flxT2. sowflisf_cav
    return so, thetao, fwfisf

def extractTSmelt_normal(masks, basins_merged, geoms, so, thetao, fwfisf, filepath_nn_input, year_of_interest):
    # Calculate the basin numbers of any basin which contains closed cavities 
    basin_nos_temp = np.unique(basins_merged)
    count = np.zeros(len(basin_nos_temp))
    for i in range(len(basin_nos_temp)):
        count[i] = np.sum((basins_merged*masks.closed_cavities_nan) == basin_nos_temp[i])
    mask_keep_nos = count != 0
    basin_nos = basin_nos_temp[mask_keep_nos]
    # Create empty arrays for the propagated temperature and salinity (on NEMO grid)
    T_prop = np.ones(so.nav_lat.data.shape)*np.nan
    S_prop = np.ones(so.nav_lat.data.shape)*np.nan
    for kk in range(len(basin_nos)):
    #kk = 55
        basin_no = basin_nos[kk]
        mask_TS_basin = xr.where((basins_merged == basin_no) & (masks.mask_TS_profiles == 1), 1, np.nan) == 1
        try:
            thetao = thetao.rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
            so = so.rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        except:
            ()
        real_profile = thetao.where(mask_TS_basin == 1).mean(dim = ['x_grid_T','y_grid_T'], skipna = True)
        T_profile = real_profile.interpolate_na(dim = 'deptht').ffill(dim = 'deptht').bfill(dim = 'deptht')
        real_profile = so.where(mask_TS_basin == 1).mean(dim = ['x_grid_T','y_grid_T'], skipna = True)
        S_profile = real_profile.interpolate_na(dim = 'deptht').ffill(dim = 'deptht').bfill(dim = 'deptht')
        mask_basin = basins_merged*masks.closed_cavities_nan == basin_no
        T_prop[mask_basin] = T_profile.sel(deptht = geoms.isf_draft.data[mask_basin], method = 'nearest')
        S_prop[mask_basin] = S_profile.sel(deptht = geoms.isf_draft.data[mask_basin], method = 'nearest')
        print('Currently processing:', year_of_interest, ',', \
                  kk+1, 'out of', len(basin_nos), 'profiles processed', end = '\r')
    # And save processed output 
    ds = xr.Dataset(
                data_vars=dict(
                    temperature_prop    = (["y", "x"], T_prop.data),
                    salinity_prop       = (["y", "x"], S_prop.data),
                    melt_ice_per_yr     = (["y", "x"], (fwfisf.values*masks.closed_cavities_nan).data),
                    ),
                coords=dict(
                    lon=(("y", "x"), so.nav_lon.data),
                    lat=(("y", "x"), so.nav_lat.data),    ),
                attrs=dict(description = "Pointwise T, S, and melt", 
                           simulation_run = 'Christoph_CFG',
                           year = year_of_interest)
                    )
    fp_TSmelt = filepath_nn_input + nemo_run +'_y' + str(year_of_interest) + '.nc'
    ds.to_netcdf(fp_TSmelt)
    print('You have saved the following file')
    print(fp_TSmelt)

filepaths = glob.glob(filepath_nemo + file_extension)
print(filepaths[0])

years = []
if monthly == True:
    months = []
if monthly == False:
    if nemo_run in ('OPM016', 'OPM018', 'OPM021'):
        for i in range(len(filepaths)):
            year = filepaths[i].split(os.path.sep)[-1].split('_')[1].split('.')[0].split('y')[1]
            years.append(year)
    if nemo_run in ('ctrl94', 'isf94', 'isfru94'):
        yr0, yrfin = filepaths[0].split(os.path.sep)[-1].split('_')[1].split('.')[0].split('-')
        yr0, yrfin
        yrlist = np.arange(int(yr0), int(yrfin)+1, 1)
        for i in range(len(yrlist)):
            years.append(yrlist[i])
years.sort()
print(len(np.unique(years)), 'files to be processed')

# Load the masks and geoms needed 
masks, basins_merged, geoms = load_geom_files(filepath_geomvars, filepath_mask, join_ice_shelves = True)
   
#for i in range(len(filepaths)):
for i in range(int(year_min), int(year_max)):
    year_of_interest = i
    # Record start time    
    start_time = time.time()
    print('Currently processing:', year_of_interest, '                                  ', end = '\r')
    if nemo_run in ('OPM016', 'OPM018', 'OPM021'):
        # Load so, thetao, fwfisf
        so, thetao, fwfisf = load_TSmelt_normal(filepath_nemo, nemo_run, year_of_interest)
        # And process the year of interest 
        extractTSmelt_normal(masks, basins_merged, geoms, so, thetao, fwfisf, filepath_nn_input, year_of_interest)
    elif nemo_run in ('ctrl94', 'isf94', 'isfru94'):
        # Load so, thetao, fwfisf
        so, thetao, fwfisf = load_TSmelt_chris(filepath_nemo, nemo_run, year_of_interest)
        # And process the year of interest 
        extractTSmelt_normal(masks, basins_merged, geoms, so, thetao, fwfisf, filepath_nn_input, year_of_interest)
    # Record end time
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f" {year_of_interest} : {elapsed_time:.2f} seconds                                      ")
print('All requested years saved to netcdf    ')  
