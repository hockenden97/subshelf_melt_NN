import numpy as np
import xarray as xr    
import glob
import pandas as pd
import itertools
import sklearn
import sys 

# Set all the filepaths that will be required 
filepath_data_ho = "/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/data/processing_ho/"
filepath_masks = filepath_data_ho + "masks_"

filepath_nn_input = filepath_data_ho + "nn_input_"
filepath_areas = "/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/Masks/areas.nc"

# Set where you would like to save the data 
data_out_fp = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/Training_data/'

this_collection = sys.argv[1]#'OPM026_OPM0263_OPM031_Christoph'

# Split the nameto call the individual datasets 
collections = this_collection.split('_')
print(collections)
# Set the filepath for each individual dataset
filepaths = []
for i in range(len(collections)):
    if collections[i] == 'Christoph':
        collections[i] = 'Christoph_v2'
    filepaths.append(data_out_fp + collections[i] + '_' + 'whole_dataset' + '_' + 'not_yet_normalised.csv')

# Load in the datasets (Set fast_check_version to True if you want to just check that this runs okay, otherwise it should be false)
fast_check_version = False
if fast_check_version == True:
    df_total = pd.read_csv(filepaths[0], nrows = 5)
    print('You have loaded', filepaths[0])
    for i in range(len(filepaths)-1):
        df_total = pd.concat([df_total, pd.read_csv(filepaths[i+1], nrows = 5)])
        print('You have loaded', filepaths[i+1])
    print('There are', df_total.shape[0], 'entries in the combined dataset')
elif fast_check_version == False:
    df_total = pd.read_csv(filepaths[0])
    print('You have loaded', filepaths[0])
    for i in range(len(filepaths)-1):
        df_total = pd.concat([df_total, pd.read_csv(filepaths[i+1])])
        print('You have loaded', filepaths[i+1])
    print('There are', df_total.shape[0], 'entries in the combined dataset')

annual_train = False
if annual_train == True:
    df_grouped = df_total.groupby(['lat','lon','year'], as_index = False).mean()
    df_total = df_grouped
    annual_f = 'annual_'
else:   
    annual_f = ''

plot_chosen_sims = False
if plot_chosen_sims == True:
    import matplotlib.pyplot as plt
    
    # Make a list of all the month/year combinations 
    year_and_months_t = np.unique(list(zip(df_all.year, df_all.month)), axis = 0)
    yr_t = year_and_months_t[:,0]
    mn_t = year_and_months_t[:,1]
    # Make a list of the requested months/years combination has been created 
    year_and_months = np.unique(list(zip(df_sel_year.year, df_sel_year.month)), axis = 0)
    yr_check = year_and_months[:,0]
    mn_check = year_and_months[:,1]
    # Create a mask for the month/years which have been left out
    mask = np.isin(year_and_months_t, year_and_months).all(axis = 1)
    # Create a random mask for the months/years which are used as validation
    a = np.zeros(yr_check.shape[0], dtype=int)
    a[:int(yr_check.shape[0]/10)] = 1
    np.random.shuffle(a)
    random_mask = a.astype(bool)

    # Plot this distribution of simulations
    fig, ax = plt.subplots(1,1, figsize = (8,2))
    plt.scatter(yr_check, mn_check, marker = "s", label = 'Training', color = 'C9')
    plt.scatter(yr_check[random_mask], mn_check[random_mask], marker = 's', color = 'C0', label = 'Validation')
    plt.scatter(yr_t[~mask], mn_t[~mask], marker = "s", color = 'C1', label = 'Testing', zorder = 0)
    ax.set_xticks((1980,1990,2000,2010,2020,2030,2040,2050,2060));
    ax.set_yticks((1,2,3,4,5,6,7,8,9,10,11,12));
    ax.set_yticklabels(('J','F','M','A','M','J','J','A','S','O','N','D'), fontsize = 8)
    #ax.set_xlim(1978,2024)
    ax.set_ylim(0,13)
    ax.set_ylabel('Month', fontweight = 'bold')
    ax.set_xlabel('Year', fontweight = 'bold')
    ax.set_title(title, fontweight = 'bold');

# Split the training and validation datasets 
fraction_for_validation = 1/9 # How much of the training/validation dataset to use for validation
                              # Assuming a 80:10:10 training:validation:testing split, set this to 1/9 
                              #    (as 1/10 for testing has already been taken)
                              # You can set this to False to use the whole dataset for training and validation
if fraction_for_validation == False:
    train_input_df1 = df_total.copy()
    val_input_df1 = df_total.copy()
    no_train = df_total.shape[0]
    no_val = df_total.shape[0]
    print('Warning: You are not creating a separate validation dataset, this may affect the robustmess of your results')
else:
    # Split the data with the desired ratio (it is also shuffled)
    train_input_df1, val_input_df1 = \
            sklearn.model_selection.train_test_split(df_total, test_size = fraction_for_validation, random_state = 1)
    no_train = train_input_df1.shape[0]
    no_val = val_input_df1.shape[0]
no_test = df_total.shape[0] - no_train - no_val
print('Training data:   {} points, {:.0f}% of data'.format(no_train, no_train*100/df_total.shape[0]))
print('Validation data: {} points,  {:.0f}% of data'.format(no_val, no_val*100/df_total.shape[0]))
print('Testing data:    {} points,  {:.0f}% of data'.format(no_test, no_test*100/df_total.shape[0]))

train_input_df = train_input_df1.to_xarray()
val_input_df = val_input_df1.to_xarray()

## Prepare the training and validation datasets
y_train = train_input_df['melt_m_ice_per_y']
x_train = train_input_df.drop_vars(['melt_m_ice_per_y'])
y_val = val_input_df['melt_m_ice_per_y']
x_val = val_input_df.drop_vars(['melt_m_ice_per_y'])
print()
print('Training and validation datasets (x and y) prepared')

def compute_norm_metrics(x_train, y_train, norm_method):
    # Calculate the mean
    x_mean = x_train.mean()
    y_mean = y_train.mean()
    # Calulate the normalisation factor 
    if norm_method == 'std':
        x_range  = x_train.std()
        y_range  = y_train.std()
    elif norm_method == 'interquart':
        x_range  = x_train.quantile(0.9) - x_train.quantile(0.1)
        y_range  = y_train.quantile(0.9) - y_train.quantile(0.1)
    elif norm_method == 'minmax':
        x_range  = x_train.max() - x_train.min() 
        y_range  = y_train.max() - y_train.min() 
    # Merge methods 
    norm_mean = xr.merge([x_mean,y_mean]).assign_coords({'metric': 'mean_vars', 'norm_method': norm_method})
    norm_range = xr.merge([x_range,y_range]).assign_coords({'metric': 'range_vars', 'norm_method': norm_method})
    # Create array of metrics
    summary_metrics = xr.concat([norm_mean, norm_range], dim='metric').assign_coords({'norm_method': norm_method})
    return summary_metrics

print(this_collection)

# Normalise the input and output data
norm_summary_list = []
for norm_method in ['std','interquart','minmax']:
    summary_ds = compute_norm_metrics(x_train, y_train, norm_method)
    norm_summary_list.append(summary_ds)
summary_ds_all = xr.concat(norm_summary_list, dim='norm_method')
print('Data normalised for all three methods')

# Calculate var mean, var 
var_mean = summary_ds_all.sel(metric='mean_vars')
var_range = summary_ds_all.sel(metric='range_vars')
var_train_norm = (train_input_df - var_mean)/var_range
var_val_norm = (val_input_df - var_mean)/var_range
print('Normalisation metrics calculated')

#set filenames
fp_metrics = data_out_fp + this_collection + '_' + annual_f + 'metrics_norm.nc'
fp_var_train_norm = data_out_fp + this_collection + '_' + annual_f + 'train_data.nc'
fp_var_val_norm = data_out_fp + this_collection + '_' + annual_f + 'val_data.nc'
#print(fp_metrics)
#print(fp_var_train_norm)
#print(fp_var_val_norm)

# Set data to variables and save to file
metrics_ds, var_train_norm, var_val_norm = summary_ds_all, var_train_norm, var_val_norm
metrics_ds.to_netcdf(fp_metrics)
var_train_norm.to_netcdf(fp_var_train_norm)
var_val_norm.to_netcdf(fp_var_val_norm)    
print('You have saved:')
print(fp_metrics)
print(fp_var_train_norm)
print(fp_var_val_norm)
