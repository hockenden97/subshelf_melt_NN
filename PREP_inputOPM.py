# Import the relevant packages
import numpy as np
import xarray as xr    
# To interpolate NEMO data from Nico's mask files to the NEMO simulation grid 
from scipy.interpolate import griddata
import itertools
# To convert coordinates from lat, lon to polar stereo x,y 
from pyproj import Transformer
# KD Tree for nearest neighbours
from scipy.spatial import cKDTree
# To play a sound when at the end of the code you want to run 
import os
##os.system("printf '\a'")
import time
import glob
import sys

def make_masks_monthly_func(nemo_run, filepath_raw_NEMO, year_of_interest, month_of_interest, \
                            filepath_masks, filepath_dsREDCAV, filepath_basin_nos, \
                            skip_interpolation = True, need_to_calculate_distances = False):
    """ This function will take one run of the NEMO simulation and create masks for """
    """ where there are open and closed cavities, and ocean and grounded ice.       """
    """ It also calculates the basin numbers, and can do the grid distances         """
    """ It requires one run for NEMO 1 degree for comparison                        """
    # Set filepath to the specified year 
    if nemo_run == 'OPM018':
        filepath_gridT_varofint = filepath_raw_NEMO + 'eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                '.1y_gridT_varofint.nc'
    else:
        filepath_gridT_varofint = filepath_raw_NEMO + str(year_of_interest) + '/eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                'm' + month_of_interest + '.1m_gridT.nc'
    # Load in the simulation files 
    gridT_varofint = xr.open_dataset(filepath_gridT_varofint)
    # Remove the uncessary axes 
    #gridT_varofint = gridT_varofint.squeeze('time_counter').sel(bnds=1).drop('time_counter_bnds').drop('time_counter')
    gridT_varofint = gridT_varofint.squeeze('time_counter').sel(axis_nbounds=1).drop('time_centered').drop('time_counter').sel(nvertex=0)
    # Crop out most of the world and just save Antarctica 
    # Specify the coordinates needed
    min_lat = -51
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    # Take this slice 
    gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))
    # Close the original dataset
    gridT_varofint.close()
    if skip_interpolation == False:
        # Open Nico's data to do the interpolation
        ds0=xr.open_dataset(filepath_basin_nos)
        dsREDCAV=xr.open_dataset(filepath_dsREDCAV)
        dsREDCAV = dsREDCAV.squeeze('time_counter')
        # Interpolate Nico's data onto the grid 
        x_flat = np.ndarray.ravel(dsREDCAV.nav_lon.data)
        y_flat = np.ndarray.ravel(dsREDCAV.nav_lat.data)
        z_flat = np.ndarray.ravel(ds0.basin.data)
        dsREDCAV_flat = np.ndarray.ravel(dsREDCAV.top_level.data)
        ds0.close()
        dsREDCAV.close()
        basins_NEMO = griddata((x_flat,y_flat), z_flat, (gridT2.nav_lon, gridT2.nav_lat), method = 'nearest')
        dsREDCAV_NEMO = griddata((x_flat,y_flat), dsREDCAV_flat, (gridT2.nav_lon, gridT2.nav_lat), method = 'nearest')
        # Save these arrays to an xarray dataset to load in later
        ds = xr.Dataset(
        data_vars=dict(
            dsREDCAV_NEMO=(["y", "x"], dsREDCAV_NEMO),
            basins_NEMO=(["y", "x"], basins_NEMO),    ),
        coords=dict(
            lon=(("y", "x"), gridT2.nav_lon.data),
            lat=(("y", "x"), gridT2.nav_lat.data),    ),
        attrs=dict(description="Basin numbers and limit of cavity resolutions", 
                   simulation_run = nemo_run),
        )
        # Save these pre-interpolated grids
        ds.to_netcdf(filepath_nico_on_nemo +nemo_run +'.nc')
    else:
        # Load the pre-interpolated grids from file
        ds_load = xr.open_dataset(filepath_nico_on_nemo+nemo_run + '.nc')
        dsREDCAV_NEMO = ds_load.dsREDCAV_NEMO.data
        basins_NEMO = ds_load.basins_NEMO.data
        ds_load.close()
    # Calculate the depth of the ice-ocean interface 
    mask_temp_values = xr.where(gridT2.votemper > -200, 1,0)
    mask_diff_temp = (mask_temp_values - mask_temp_values.shift(deptht=1))
    positive = xr.where(mask_diff_temp == 1, 1, np.nan)
    positive_10 = xr.where(mask_diff_temp == 1, 1, 0)
    negative = xr.where(mask_diff_temp == -1, 1, np.nan)
    negative_10 = xr.where(mask_diff_temp == -1, 1, 0)
    # Create the masks for the regions which will be of interest later 
    # All the cavities
    all_cavities = positive_10.sum(dim = 'deptht')
    all_cavities_nan = xr.where(all_cavities == 1, 1, np.nan)
    # The ocean and all the cavities 
    ocean_all_cavities = negative_10.sum(dim = 'deptht')
    ocean_all_cavities_nan = xr.where(ocean_all_cavities == 1, 1, np.nan)
    # Grounded ice (not the ocean or a cavity)
    grounded_ice = 1 - ocean_all_cavities 
    grounded_ice_nan = xr.where(grounded_ice == 1, 1, np.nan)
    # Only the ocean 
    ocean_only = ocean_all_cavities - all_cavities
    ocean_only_nan = xr.where(ocean_only == 1, 1, np.nan)
    # Only cavities which are open in NEMO 1 degree 
    open_cavities_mask = dsREDCAV_NEMO > 1
    open_cavities = np.ones(open_cavities_mask.shape)
    open_cavities_nan = np.ones(open_cavities_mask.shape)
    open_cavities[open_cavities_mask == False] = 0
    open_cavities_nan[open_cavities_mask == False] = np.nan
    # Cavities which are open in NEMO 0.25 degree but not NEMO 1 degree 
    closed_cavities = all_cavities - open_cavities
    closed_cavities_nan = xr.where(closed_cavities == 1, 1, np.nan)
    # The ocean and oepn cavities (to calculate the local ice shelf front) 
    ocean_open_cavities = ocean_only + open_cavities
    ocean_open_cavities_nan = xr.where(ocean_open_cavities == 1, 1, np.nan)
    # Create an xr Dataset to save these variables to 
    ds = xr.Dataset(
        data_vars=dict(
            all_cavities=(["y", "x"], all_cavities.data),
            all_cavities_nan=(["y", "x"], all_cavities_nan.data),
            ocean_all_cavities=(["y", "x"], ocean_all_cavities.data),
            ocean_all_cavities_nan=(["y", "x"], ocean_all_cavities_nan.data),
            ocean_only=(["y", "x"], ocean_only.data),
            ocean_only_nan=(["y", "x"], ocean_only_nan.data),
            open_cavities=(["y", "x"], open_cavities.data),
            open_cavities_nan=(["y", "x"], open_cavities_nan.data),
            closed_cavities=(["y", "x"], closed_cavities.data),
            closed_cavities_nan=(["y", "x"], closed_cavities_nan.data),
            ocean_open_cavities=(["y", "x"], ocean_open_cavities.data),
            ocean_open_cavities_nan=(["y", "x"], ocean_open_cavities_nan.data),
            grounded_ice=(["y", "x"], grounded_ice.data),
            grounded_ice_nan=(["y", "x"], grounded_ice_nan.data),    ),
        coords=dict(
            lon=(("y", "x"), gridT2.nav_lon.data),
            lat=(("y", "x"), gridT2.nav_lat.data),    ),
        attrs=dict(description="Masks for grounded ice, floating ice and open ocean", 
                   simulation_run = nemo_run),
    )
    # Save the variable to netcdf
    ds.to_netcdf(filepath_masks + nemo_run + '.nc')
    print('Masks saved to netcdf')
    if need_to_calculate_distances == True:
        # In order to calculate the slopes we need the distance between points, 
        # and because the points are on the NEMO lat-lon grid we need to convert them to polar stereo the calculate the distances 
        # Create a pyproj transformer to transform onto polar stereo
        transformer = Transformer.from_crs("epsg:4326","epsg:3031")
        lat, lon = gridT2.nav_lat.data, gridT2.nav_lon.data
        grid_x, grid_y = transformer.transform(lat, lon)
        # Use the np.roll function to move the data slightly 
        shifted_x_p0 = (grid_x - np.roll(grid_x, shift = 1, axis = 0))
        shifted_x_m0 = (grid_x - np.roll(grid_x, shift = -1, axis = 0))
        shifted_x_p1 = (grid_x - np.roll(grid_x, shift = 1, axis = 1))
        shifted_x_m1 = (grid_x - np.roll(grid_x, shift = -1, axis = 1))
        shifted_y_p0 = (grid_y - np.roll(grid_y, shift = 1, axis = 0))
        shifted_y_m0 = (grid_y - np.roll(grid_y, shift = -1, axis = 0))
        shifted_y_p1 = (grid_y - np.roll(grid_y, shift = 1, axis = 1))
        shifted_y_m1 = (grid_y - np.roll(grid_y, shift = -1, axis = 1))
        # Because you need to calculate the hypotenuse distance you need x and y for both directions of shift 
        shifted_p_lon = np.sqrt(shifted_x_p0**2 + shifted_y_p0**2) # lon
        shifted_m_lon = np.sqrt(shifted_x_m0**2 + shifted_y_m0**2) # lon 
        shifted_p_lat = np.sqrt(shifted_x_p1**2 + shifted_y_p1**2) # lat
        shifted_m_lat = np.sqrt(shifted_x_m1**2 + shifted_y_m1**2) # lat
        # And then save these to file 
        ds = xr.Dataset(
            data_vars=dict(
                dist_p_lon=(["y", "x"], shifted_p_lon),
                dist_m_lon=(["y", "x"], shifted_m_lon),
                dist_p_lat=(["y", "x"], shifted_p_lat),
                dist_m_lat=(["y", "x"], shifted_m_lat),    ),
            coords=dict(
                lon=(("y", "x"), gridT2.nav_lon.data),
                lat=(("y", "x"), gridT2.nav_lat.data),    ),
            attrs=dict(description="Distances", 
                       simulation_run = nemo_run),
        )
        print(filepath_distances)
        #ds.to_netcdf(filepath_distances)
        print('Distances saved to netcdf')

def extract_useful_variables_monthly(nemo_run, year_of_interest, month_of_interest, filepath_raw_NEMO, filepath_masks, 
                                     filepath_distances, filepath_nn_inputs, verbose = 0):
    if nemo_run == 'OPM018':
        filepath_gridT_varofint = filepath_raw_NEMO + 'eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                    '.1y_gridT_varofint.nc'
        filepath_flxT_varofint = filepath_raw_NEMO + 'eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                    '.1y_flxT_varofint.nc'
    else:
        filepath_gridT_varofint = filepath_raw_NEMO + str(year_of_interest) + '/eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                    'm' + month_of_interest + '.1m_gridT.nc'
        filepath_flxT_varofint = filepath_raw_NEMO + str(year_of_interest) + '/eORCA025.L121-' + nemo_run + '_y' + str(year_of_interest) + \
                                    'm' + month_of_interest + '.1m_flxT.nc'
    # Load in the simulation files 
    gridT_varofint = xr.open_dataset(filepath_gridT_varofint)
    # votemper = sea_water_potential_temperature
    # vosaline = sea_water_practical_salinity
    # sosbt = sea bottom temperature
    # sosbs = sea bottom salinity
    # sosst = sea surface temperature
    # sosss = sea surface salinity 
    flxT_varofint = xr.open_dataset(filepath_flxT_varofint)
    # sowflisf_cav ice shelf melt rate in kg/m2/s
    # Cut out variables which are not needed 
    # Cut out variables which are not needed 
    if nemo_run == 'OPM018':
        gridT_varofint = gridT_varofint.squeeze('time_counter').sel(bnds=1).drop('time_counter')
        flxT_varofint = flxT_varofint.squeeze('time_counter').sel(bnds=1).drop('time_counter')
    else:
        gridT_varofint = gridT_varofint.squeeze('time_counter').sel(axis_nbounds=1).drop('time_centered').drop('time_counter').sel(nvertex=0)
        flxT_varofint = flxT_varofint.squeeze('time_counter').sel(axis_nbounds=1).drop('time_centered').drop('time_counter').sel(nvertex=0)
# Crop out most of the world and just save Antarctica 
    # Specify the coordinates needed
    min_lat = -51
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    # Take this slice 
    gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))
    flxT2 = flxT_varofint.sel(y = slice(0,y_trim+1))
    # Close the original dataset
    gridT_varofint.close()
    flxT_varofint.close()
    if verbose == 2:
        print('Data loaded and cropped')
    # The melt rate comes directly from flxT2.sowflisf_cav (ice shelf melt rate in kg/m2/s)
    # Calculate the distance to the grounding line, distance to the local ice front, and distance to the far away ice front
    # Load in masks 
    masks = xr.open_dataset(filepath_masks + nemo_run + '.nc')
    ocean_open_cavities = masks.ocean_open_cavities.data
    grounded_ice = masks.grounded_ice.data
    ocean_only = masks.ocean_only.data
    closed_cavities = masks.closed_cavities.data
    masks.close()
    # Create pyproj transformer to transform from lat lon to polar sterographic 
    transformer = Transformer.from_crs("epsg:4326","epsg:3031")
    # Create a list of all the grounded ice points
    GL_lat, GL_lon = gridT2.nav_lat.data[grounded_ice == 1], gridT2.nav_lon.data[grounded_ice == 1]
    GL_x, GL_y = transformer.transform(GL_lat, GL_lon)
    points_GL = np.array(list(zip(GL_x, GL_y)))
    # Create a list of all the points in closed ice shelves 
    CC_lat, CC_lon = gridT2.nav_lat.data[closed_cavities == 1], gridT2.nav_lon.data[closed_cavities == 1]
    CC_x, CC_y = transformer.transform(CC_lat, CC_lon)
    points_CC = np.array(list(zip(CC_x, CC_y)))
    # Create the cKD Tree for the grounding line 
    tree = cKDTree(points_GL)
    # Query the cKD Tree for the closed cavity points 
    dist, index = tree.query(points_CC, k = 1)
    distances_GL = np.ones(gridT2.nav_lat.data.shape)*np.nan
    distances_GL[closed_cavities == 1] = dist
    # Create a list of all the open ocean points 
    OO_lat, OO_lon = gridT2.nav_lat.data[ocean_only == 1], gridT2.nav_lon.data[ocean_only == 1]
    OO_x, OO_y = transformer.transform(OO_lat, OO_lon)
    points_OO = np.array(list(zip(OO_x, OO_y)))
    # Create the cKD Tree for the open ocean
    tree = cKDTree(points_OO)
    # Query the cKD Tree for the closed cavity points 
    dist, index = tree.query(points_CC, k = 1)
    distances_OO = np.ones(gridT2.nav_lat.data.shape)*np.nan
    distances_OO[closed_cavities == 1] = dist
    if verbose == 2:
        print('Some distances calculated')
    # To calculate the distance to the nearest local ocean point (this is needed for the T_S profiles too)
    # There are complications because the nearest ocean point is not necessarily the nearest connected ocean point 
    # Load in the basins mask
    nico_on_nemo = xr.open_dataset(filepath_nico_on_nemo + nemo_run + '.nc')
    basins_NEMO = nico_on_nemo.basins_NEMO.data
    nico_on_nemo.close()
    # Some ice shelves which should potentially be joined together into one bigger ice shelf
    join_ice_shelves = True
    if join_ice_shelves == True:
        # Dotson and Crosson?
        #basins_NEMO[basins_NEMO == 101] = 129
        # Abbot Ice Shelf
        basins_NEMO[basins_NEMO == 109] = 143
        # George VI
        basins_NEMO[basins_NEMO == 112] = 125
        # Lambert 
        basins_NEMO[basins_NEMO == 20] = 103
    # Calculate the basin numbers of any basin which contains closed cavities 
    basin_nos_temp = np.unique(basins_NEMO)
    count = np.zeros(len(basin_nos_temp))
    for i in range(len(basin_nos_temp)):
        count[i] = np.sum((basins_NEMO*masks.closed_cavities_nan) == basin_nos_temp[i])
    mask_keep_nos = count != 0
    basin_nos = basin_nos_temp[mask_keep_nos]
    # Create a list of all the open ocean and open cavity points 
    OC_lat, OC_lon = gridT2.nav_lat.data[ocean_open_cavities == 1], gridT2.nav_lon.data[ocean_open_cavities == 1]
    OC_x, OC_y = transformer.transform(OC_lat, OC_lon)
    points_OC = np.array(list(zip(OC_x, OC_y)))
    # Create a cKD Tree for the ocean points 
    tree = cKDTree(points_OC)
    # Calculate the depth of the ice-ocean interface 
    mask_temp_values = xr.where(gridT2.votemper > -200, 1,0)
    mask_diff_temp = (mask_temp_values - mask_temp_values.shift(deptht=1))
    positive = xr.where(mask_diff_temp == 1, 1, np.nan)
    negative = xr.where(mask_diff_temp == -1, 1, np.nan)
    # Create an empty array for the local distances 
    distances_OC = np.ones(gridT2.nav_lat.data.shape)*np.nan
    # Create empty arrays for the propagated temperature and salinity (on NEMO grid)
    T_prop = np.ones(gridT2.nav_lat.data.shape)*np.nan
    S_prop = np.ones(gridT2.nav_lat.data.shape)*np.nan
    # Create arrays for the temperature and salinity profiles
    T_profiles = np.ndarray(len(basin_nos), dtype = np.ndarray) 
    S_profiles = np.ndarray(len(basin_nos), dtype = np.ndarray) 
    # Create an array for the direction to the front of the closed ice shelf cavity 
    direction_to_front = np.ones(gridT2.nav_lat.data.shape)*np.nan
    # Iterate over all the basins 
    for kk in range(len(basin_nos)):
    #for kk in (1,31):
        i = basin_nos[kk]
        # Extract the closed cavity points which are in that basin
        mask_basin = basins_NEMO*masks.closed_cavities_nan == i
        x_b, y_b = transformer.transform(gridT2.nav_lat.data[mask_basin], gridT2.nav_lon.data[mask_basin])
        points_basin = np.array(list(zip(x_b,y_b)))
        # Query the cKDTree to calculate the distances to points within the basin
        distance, index = tree.query(points_basin)
        # Extract only the points that are adjacent to the closed cavity (by using the minimum distance) 
        results = tree.query_ball_point(points_basin, r = np.min(distance)*np.sqrt(2))
        # Extract indexes and create a list of the unique points adjacent to the closed cavity 
        merged = np.unique(list(itertools.chain.from_iterable(results)))
        chosen_x = OC_x[merged]
        chosen_y = OC_y[merged]
        chosen_x, chosen_y
        points_chosen = np.array(list(zip(chosen_x, chosen_y)))
        # Create a cKD Tree with the chosen points 
        tree_i = cKDTree(points_chosen)
        # Query the tree for the points within the closed cavity
        distance, index = tree_i.query(points_basin)
        distances_OC[mask_basin] = distance
        chosen_lon = flxT2.nav_lon.data[ocean_open_cavities == 1][merged]
        chosen_lat = flxT2.nav_lat.data[ocean_open_cavities == 1][merged]
        # Calculate the average direction to the edge of the closed ice shelf cavity 
        x_mdiff = np.mean(chosen_lon) - np.mean(gridT2.nav_lon.data[mask_basin])
        y_mdiff = np.mean(chosen_lat) - np.mean(gridT2.nav_lat.data[mask_basin])
        direction = np.arctan2(y_mdiff, x_mdiff)
        direction_to_front[mask_basin] = np.ones(len(points_basin)) * direction
        # Create a mask of the chosen points along the front of the cavity 
        a = xr.where(gridT2.nav_lon == 200,0,0)
        for j in range(len(chosen_lon)):
            a =a + xr.where((gridT2.nav_lat == chosen_lat[j]) & (gridT2.nav_lon == chosen_lon[j]),1, 0).data
        # Use this mask to select the points of interest and then take the mean for each depth 
        T_profiles[kk] = gridT2.votemper.where(a == 1).mean(dim = ['x','y'])
        S_profiles[kk] = gridT2.vosaline.where(a == 1).mean(dim = ['x','y'])
        # Calculate the depth of the ice-ocean interface 
        depths_of_interest = (positive*gridT2.deptht).sum(dim = 'deptht').data[mask_basin]
        T_profile = T_profiles[kk]
        S_profile = S_profiles[kk]
        # Set the T and S profiles so that they have values at all depths from the top and bottom of the bit with values 
        mask = T_profile > -20 # 
        # Calculate max and min ocean profile depths and associated temperatures 
        max_profile_depth = np.max(T_profile.deptht[mask].data)
        max_profile_depth_T = T_profile.sel(deptht = max_profile_depth)
        max_profile_depth_S = S_profile.sel(deptht = max_profile_depth)
        min_profile_depth = np.min(T_profile.deptht[mask].data)
        min_profile_depth_T = T_profile.sel(deptht = min_profile_depth)
        min_profile_depth_S = S_profile.sel(deptht = min_profile_depth)
        mask_top = (T_profile.deptht <= min_profile_depth)
        mask_bottom = (T_profile.deptht > max_profile_depth)
        # Propagate the profiles with these values 
        T_profile_np = T_profile.data
        T_profile_np[mask_top] = min_profile_depth_T
        T_profile_np[mask_bottom] = max_profile_depth_T
        S_profile_np = S_profile.data
        S_profile_np[mask_top] = min_profile_depth_S
        S_profile_np[mask_bottom] = max_profile_depth_S
        # Create arrays for the data to be propagated to
        T_of_interest = np.ndarray(len(depths_of_interest))
        S_of_interest = np.ndarray(len(depths_of_interest))
        # propagate the data to these arrays 
        for jj in range(len(depths_of_interest)):
            T_of_interest[jj] = T_profile.sel(deptht = depths_of_interest[jj], method = 'nearest')
            S_of_interest[jj] = S_profile.sel(deptht = depths_of_interest[jj], method = 'nearest')
        T_prop[mask_basin] = T_of_interest
        S_prop[mask_basin] = S_of_interest
        if verbose == 2:
            print('Creating and propagating profiles:', kk+1, 'out of', len(basin_nos), 'processed', end = '\r')
        if verbose == 1:
            print('Currently processing:', year_of_interest, month_of_interest, ',', \
                  kk+1, 'out of', len(basin_nos), 'profiles processed', end = '\r')
    if verbose == 2:
        print('Propagated profiles completed                             ')
    # Calculate the depth of the interface and the bathymetry
    interface_depth = (positive*gridT2.deptht).sum(dim = 'deptht')#*masks.closed_cavities_nan
    bathymetric_depth = (negative*gridT2.deptht).sum(dim = 'deptht')#*masks.closed_cavities_nan
    if verbose == 2: 
        print('Interface depth and bathymetric depth')
    # Calculate the local slopes in all 4 directions 
    distances_grid = xr.open_dataset(filepath_distances)
    interface = interface_depth.data*masks.all_cavities_nan
    interface_p_lon = ( + interface - np.roll(interface, shift = 1, axis = 0))/distances_grid.dist_p_lon
    interface_m_lon = (- interface + np.roll(interface, shift = -1, axis = 0))/distances_grid.dist_m_lon
    interface_p_lat = (+ interface - np.roll(interface, shift = 1, axis = 1))/distances_grid.dist_p_lat
    interface_m_lat = (- interface + np.roll(interface, shift = -1, axis = 1))/distances_grid.dist_m_lat
    bathymetry = bathymetric_depth.data*masks.all_cavities_nan
    bathymetry_p_lon = ( + bathymetry - np.roll(bathymetry, shift = 1, axis = 0))/distances_grid.dist_p_lon
    bathymetry_m_lon = (- bathymetry + np.roll(bathymetry, shift = -1, axis = 0))/distances_grid.dist_m_lon
    bathymetry_p_lat = (+ bathymetry - np.roll(bathymetry, shift = 1, axis = 1))/distances_grid.dist_p_lat
    bathymetry_m_lat = (- bathymetry + np.roll(bathymetry, shift = -1, axis = 1))/distances_grid.dist_m_lat
    distances_grid.close()
    # Take the mean to get the actual local slope (and account for bendy points)
    slope_isdraft_lon = np.nanmean([interface_p_lon, interface_m_lon], axis = 0)
    slope_isdraft_lat = np.nanmean([interface_p_lat, interface_m_lat], axis = 0)
    slope_bathy_lon = np.nanmean([bathymetry_p_lon, bathymetry_m_lon], axis = 0)
    slope_bathy_lat = np.nanmean([bathymetry_p_lat, bathymetry_m_lat], axis = 0)
    if verbose == 2:
        print('Local meridional and zonal slopes')
    # Then rotate the calculated slopes to have the slope towards the ice front and the slope across the ice front 
    slope_isdraft_hypot = np.sqrt(slope_isdraft_lon **2 + slope_isdraft_lat **2)
    angle_isdraft = np.arctan2(slope_isdraft_lon,slope_isdraft_lat)- direction_to_front
    slope_is_across_front = slope_isdraft_hypot * np.sin(angle_isdraft)
    slope_is_towards_front = slope_isdraft_hypot * np.cos(angle_isdraft)
    slope_bathy_hypot = np.sqrt(slope_bathy_lon **2 + slope_bathy_lat **2)
    angle_bathy = np.arctan2(slope_bathy_lon,slope_bathy_lat)- direction_to_front
    slope_ba_across_front = slope_bathy_hypot * np.sin(angle_bathy)
    slope_ba_towards_front = slope_bathy_hypot * np.cos(angle_bathy)
    if verbose == 2:
        print('Rotated slopes')
    masks.close()
    # And finally, save all the variables to file 
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            distances_GL=(["y", "x"], distances_GL),
            distances_OO=(["y", "x"], distances_OO),
            distances_OC=(["y", "x"], distances_OC),
            area = (["y", "x"], gridT2.area.data),
            temperature_prop=(["y", "x"], T_prop),
            salinity_prop=(["y", "x"], S_prop),
            melt_ice_per_yr=(["y", "x"], flxT2.sowflisf_cav.data),
            corrected_isdraft=(["y", "x"], (interface_depth*masks.closed_cavities_nan).data),
            bathymetry=(["y", "x"], (bathymetric_depth*masks.closed_cavities_nan).data), 
            slope_is_lon =(["y", "x"], slope_isdraft_lon.data),
            slope_is_lat =(["y", "x"], slope_isdraft_lat.data), 
            slope_ba_lon =(["y", "x"], slope_bathy_lon.data),
            slope_ba_lat =(["y", "x"], slope_bathy_lat.data),
            slope_is_across_front =(["y", "x"], slope_is_across_front.data),
            slope_is_towards_front =(["y", "x"], slope_is_towards_front.data), 
            slope_ba_across_front =(["y", "x"], slope_ba_across_front.data),
            slope_ba_towards_front =(["y", "x"], slope_ba_towards_front.data), ),
        coords=dict(
            lon=(("y", "x"), gridT2.nav_lon.data),
            lat=(("y", "x"), gridT2.nav_lat.data),    ),
        attrs=dict(description="Neural network input variables", 
                   simulation_run = nemo_run, 
                   year = str(year_of_interest)),
    )
    # Export to netcdf 
    ds.to_netcdf(filepath_nn_input + nemo_run + '_' + str(year_of_interest) + '_' + month_of_interest + '.nc')
    if verbose == 2:
        print('All variables of interest saved to netcdf')

print('Welcome')
nemo_run = 'OPM031'
verify_if_already_run = False
year_to_run = sys.argv[1]
print(year_to_run)

# Set all the filepaths that will be required 
filepath_data_ho = "/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/data/processing_ho/"
filepath_masks = filepath_data_ho + "masks_"
filepath_nico_on_nemo = filepath_data_ho + "nico_on_nemo_"
filepath_nn_input = filepath_data_ho + "nn_input_"
filepath_distances = filepath_data_ho + "grid_distances.nc" 

#redone_slopes_fp = filepath_nn_input + nemo_run + '_' + 'redo_slopes2' + '.nc'

# Set the filepath to the NEMO files from Pierre
if nemo_run == 'OPM018':
    filepath_pierre = '/bettik/burgardc/DATA/BASAL_MELT_PARAM/raw/NEMO_eORCA025.L121-OPM018/'
    file_extension = '*gridT*.nc'
elif nemo_run == 'OPM026':
    filepath_pierre = '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPM026/eORCA025.L121-OPM026-S/1m/'
    file_extension = '*gridT*.nc'
elif nemo_run == 'OPM0261':
    filepath_pierre = '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPM0261/eORCA025.L121-OPM0261-S/1m'
    file_extension = '*/*gridT*.nc'
elif nemo_run == 'OPM0263':
    filepath_pierre = '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPM0263/1m'
    file_extension = '*/*gridT*.nc'
elif nemo_run == 'OPM031':
    filepath_pierre = '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPM031/eORCA025.L121-OPM031-S/1m/'
    file_extension = '*/*gridT.nc'
elif nemo_run == 'OPMfake':
    filepaths = ['/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPMfake/1m/2097/OPM026_1997_01_gridT_l.nc', 
                 '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPMfake/1m/2098/OPMfake_1998_02_gridT_l.nc',
                 '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-OPMfake/1m/2098/OPMfake_2099_03_gridT_l.nc']
    filepath_pierre = []
    file_extension = []
else:
    filepath_pierre = []
    file_extension = []
    print('Help, I do not know this nemo_run')
print(nemo_run, filepath_pierre + file_extension)

if nemo_run != 'OPMfake':
    # Load the files at the appropriate filepath
    filepaths = glob.glob(filepath_pierre + file_extension)
print('There are', len(filepaths), 'files')
#print(filepaths[0:3])

# From the available filepaths, work out what combination of years and months there are
simulations = []
years = []
months = []
for i in range(len(filepaths)):
    if nemo_run == 'OPM018':
        year = int(filepaths[i].split(os.path.sep)[7].split('_')[1].split('.')[0].split('y')[1])
        month = ''
        simulation = filepaths[i].split(os.path.sep)[7].split('_')[0].split('-')[1]
    elif nemo_run == 'OPM026':
        year = int(filepaths[i].split(os.path.sep)[9].split('_')[1].split('.')[0].split('m')[0].split('y')[1])
        month = (filepaths[i].split(os.path.sep)[9].split('_')[1].split('.')[0].split('m')[1])
        simulation = filepaths[i].split(os.path.sep)[9].split('_')[0].split('-')[1]
    elif nemo_run == 'OPM031':
        year = int(filepaths[i].split(os.path.sep)[10].split('_')[1].split('.')[0].split('m')[0].split('y')[1])
        month = (filepaths[i].split(os.path.sep)[10].split('_')[1].split('.')[0].split('m')[1])
        simulation = filepaths[i].split(os.path.sep)[10].split('_')[0].split('-')[1]
    elif nemo_run == 'OPMfake':
        year = int(filepaths[i].split(os.path.sep)[9].split('_')[1])
        month = filepaths[i].split(os.path.sep)[9].split('_')[2]
        simulation = filepaths[i].split(os.path.sep)[9].split('_')[0]
    else:
        print('Help, I do not know this nemo_run')
    years.append(year)
    months.append(month)
    simulations.append(simulation)

# Check if these years and months have already been run?
if verify_if_already_run == True:
    years2 = []
    months2 = []
    simulations2 = []
    # Set filepath to simulations which have already been done
    filepaths_run = glob.glob(filepath_nn_input + nemo_run + '*.nc')
    #filepaths_run = glob.glob(filepath_nn_input + 'OPM026' + '*.nc')
    filepaths_to_run = []
    for i in range(len(filepaths)):  
        if nemo_run == 'OPM018':
            fp_run = filepath_nn_input + simulations[i] + '_' + str(years[i]) + '.nc'
        else:
            fp_run = filepath_nn_input + simulations[i] + '_' + str(years[i]) + '_' + months[i] + '.nc'
        if (fp_run in filepaths_run) == False:
            filepaths_to_run.append(filepaths[i])
            years2.append(years[i])
            months2.append(months[i])
            simulations2.append(simulations[i])
    years = years2
    months = months2
    simulations = simulations2
else:
    filepaths_to_run = filepaths

print(np.unique(years))
print(len(filepaths), 'files found in folder')
print(len(filepaths_to_run), 'of these still need to be processed.')
#print(filepaths_to_run[0:3])
#print(years[0:3], months[0:3], simulations[0:3])

for i in range(len(filepaths_to_run)):
    year_of_interest = years[i]
    month_of_interest = months[i]
    nemo_run = simulations[i]
    if str(year_of_interest) == str(year_to_run):
        # Record start time    
        start_time = time.time()
        print('Currently processing:', years[i], months[i],'                           ', end = '\r')
        extract_useful_variables_monthly(nemo_run, year_of_interest, month_of_interest, filepath_pierre, \
                                         filepath_masks, filepath_distances, filepath_nn_input, verbose = 0)
        # Record end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print(f" {years[i]} {int(months[i]):02d} : {elapsed_time:.2f} seconds                         ")
print('All requested years saved to netcdf    ')  
