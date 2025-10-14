# subshelf_melt_NN
This repository contains scripts which accompany the manuscript "A neural network emulator of ice-shelf melt rates for use in low resolution ocean models". 

## Paper figures 
```
PAPER_FIGURES.ipynb  # To plot the figures from the manuscript 
PAPER_metrics.ipynb  # To calculate metrics when a neural network has been applied to a simulation
functions.py         # Some useful functions 
```

## To normalise input data for neural network training

Input data (.csv) is the processed data with columns for each of the input variables. 
(lat, lon, temperature_prop, salinity_prop, melt_m_ice_per_y, mean_T, mean_S, std_T, std_S, year, month, basins_NEMO, distances_GL, distances_OO, distances_OC, corrected_isdraft, area, bathymetry, slope_is_lon, slope_is_lat, slope_ba_lon, slope_ba_lat, slope_is_across_front, slope_is_towards_front, slope_ba_across_front, slope_ba_towards_front) 

Output data is three files, one for the normalised metrics, one for the normalised training data, and one for the validation data (all .nc) 

```
NN_norm_input_OPM_mixed_fullruns.py  # Python script to normalise input data 
NORM_inputOPM_mixed.sh               # Bash script to normalise input data 
```

## To train a neural network 

Input data is the normalised training data and normalised validation data. 

Output data is the training neural network (.keras) and the history file for the training (.history)

```
training.py          # Python script to train a neural network
TRAIN_seedloop.sh    # Bash script to train ensemble of 10 neural networks (seeds 1-10)
```

## To apply a trained neural network to simulation data 

Input data is the trained neural network (.keras) and a processed dataset for the target simulation (.csv)

Output data (.csv) is a dataset with the predicted melt rate from the neural network. 

```
APPLY_trainedNNtoOPM.py # Python script to apply a trained neural network to simulation data 
APPLY_trainedNNtoOPM.sh # Bash script to apply a trained neural network to simulation data 
```
