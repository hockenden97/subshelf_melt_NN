"""
Created on Wed Jan 25 17:41 2022

This script is to train a NN on the whole dataset

Author: Clara Burgard
"""

import numpy as np
import xarray as xr
import pandas as pd
from tqdm.notebook import trange, tqdm
import glob
import datetime
import time
import sys

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(1, '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt')

#def get_model(size,shape,activ_fct,output_shape): #'mini', 'small', 'medium', 'large', 'extra_large'
#    model = keras.models.Sequential()
#    model.add(keras.layers.Input(shape, name="InputLayer"))
#    if size == 'small':
#        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n1'))
#        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n2'))
#        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n3'))    
#    model.add(keras.layers.Dense(output_shape, name='Output'))
#    model.compile(optimizer = 'adam',
#                  loss      = 'mse',
#                  metrics   = ['mae', 'mse'] ) 
#    return model
import nn_functions.model_functions as modf

######### READ IN OPTIONS

mod_size = str(sys.argv[1]) #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt = str(sys.argv[2]) # extrap, whole, thermocline
norm_method = str(sys.argv[3]) # std, interquart, minmax
exp_name = str(sys.argv[4])
seed_nb = int(sys.argv[5])
this_collection = int(sys.argv[6])
annual_f = int(sys.argv[7])


#mod_size = 'small'
#TS_opt = 'extrap'
#norm_method = 'std'
#exp_name = 'slope_front'
#seed_nb = 1
#this_collection = 'whole_dataset' # Which dataset to use for training
#annual_f = 'annual_' # #or just empty '' if you want the whole thing 

print('Set options')
print('Experiment name is', exp_name)
print('Dataset is', this_collection)
print('Annual_only is', annual_f)

np.random.seed(seed_nb)
tf.random.set_seed(seed_nb)


if exp_name == 'all_vars':
    var_list = ['lat', 'lon', 'year', 'month',
               'distances_GL', 'distances_OO', 'distances_OC',
               'temperature_prop', 'mean_T', 'std_T',
               'salinity_prop', 'mean_S', 'std_S',
               'corrected_isdraft', 'slope_is_lon', 'slope_is_lat', 'slope_is_across_front', 'slope_is_towards_front',
               'bathymetry', 'slope_ba_lon', 'slope_ba_lat', 'slope_ba_across_front', 'slope_ba_towards_front',
               'melt_m_ice_per_y']
elif exp_name == 'slope_lat_lon':
    var_list =   ['distances_GL', 'distances_OO', 'distances_OC', 
                 'temperature_prop', 'salinity_prop', 'mean_T', 'mean_S', 'std_T', 'std_S',
                 'corrected_isdraft', 'slope_is_lon', 'slope_is_lat', 
                 'bathymetry', 'slope_ba_lon', 'slope_ba_lat',  
                 'melt_m_ice_per_y']
elif exp_name == 'slope_front':
    var_list =   ['distances_GL', 'distances_OO', 'distances_OC',
                 'temperature_prop', 'mean_T', 'std_T',
                 'salinity_prop', 'mean_S', 'std_S',
                 'corrected_isdraft', 'slope_is_across_front', 'slope_is_towards_front',
                 'bathymetry', 'slope_ba_across_front', 'slope_ba_towards_front',
                 'melt_m_ice_per_y']

print('Set var list')

######### READ IN DATA

# Filepath for normalised inputs
fp_training_data =  '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/Training_data/'
fp_var_train_norm = fp_training_data + this_collection + '_' + annual_f + 'train_data.nc'
fp_var_val_norm =   fp_training_data + this_collection + '_' + annual_f + 'val_data.nc'

# Filepath for outputs 
outputpath_nn_models = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/NN_models/'
fp_model =   outputpath_nn_models + 'model_nn_' + \
             mod_size + '_' + exp_name + '_' + this_collection + '_' + annual_f + \
             str(seed_nb).zfill(2) + '_TS' + TS_opt + '_norm' + norm_method + '.keras'
fp_history = outputpath_nn_models + 'history_' + \
             mod_size + '_' + exp_name + '_' + this_collection + '_' + annual_f + \
             str(seed_nb).zfill(2) + '_TS' + TS_opt + '_norm' + norm_method + '.csv'

if TS_opt == 'extrap':
    
    data_train_orig_norm = xr.open_dataset(fp_var_train_norm)
    data_val_orig_norm = xr.open_dataset(fp_var_val_norm) 
    print('You have loaded:')
    print(fp_var_train_norm)
    print(fp_var_val_norm)

    data_train_norm = data_train_orig_norm[var_list]
    data_val_norm = data_val_orig_norm[var_list]

    ## prepare input and target
    y_train_norm = data_train_norm['melt_m_ice_per_y'].sel(norm_method=norm_method).load()
    x_train_norm = data_train_norm.drop_vars(['melt_m_ice_per_y']).sel(norm_method=norm_method).to_array().load()
    
    y_val_norm = data_val_norm['melt_m_ice_per_y'].sel(norm_method=norm_method).load()
    x_val_norm = data_val_norm.drop_vars(['melt_m_ice_per_y']).sel(norm_method=norm_method).to_array().load()

else:
    print('Sorry, I dont know this option for TS input yet, you need to implement it...')

######### TRAIN THE MODEL

#input_size = x_train_norm.values.shape[0]
input_size = x_train_norm.values.shape[0]
activ_fct = 'relu' 
epoch_nb = 100
batch_siz = 512

input_shape=(input_size,)

model = modf.get_model(mod_size, input_shape, activ_fct,1)
print('Loaded model')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0000001, min_delta=0.0005) #, min_delta=0.1
            
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    #min_delta=0.000001,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

time_start = datetime.datetime.now()
print('Starting to fit model at:', time_start)

# Can change x_train_norm back to x_train_norm.T.values if you remove the previous 4 lines 

history = model.fit(x_train_norm.T.values,
                    y_train_norm.T.values,
                    epochs          = epoch_nb,
                    batch_size      = batch_siz,
                    verbose         = 2,
                    validation_data = (x_val_norm.T.values, y_val_norm.T.values),
                   callbacks=[reduce_lr, early_stop])

time_end = datetime.datetime.now()
print('Finished fitting model at:', time_end)
hrs, mins = divmod((time_end - time_start).seconds, 60*60)
mins, secs = divmod(mins, 60)
print('Runtime: {}:{}:{}'.format(str(hrs).zfill(2), str(mins).zfill(2), str(secs).zfill(2)))


should_i_save = True
if should_i_save == True:
    model.save(fp_model)
    print('The trained neural network has been saved to this location :')
    print(fp_model)
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    with open(fp_history, mode='w') as f:
        hist_df.to_csv(f, index = False)
    print('The history file has also been saved to csv:')
    print(fp_history)
else:
    print('Nothing has been saved yet, change should_i_save to True if you want to save')

