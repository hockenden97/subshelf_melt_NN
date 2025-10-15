# subshelf_melt_NN
This repository contains scripts which accompany the manuscript "A neural network emulator of ice-shelf melt rates for use in low resolution ocean models". 

##
##

## Paper figures 

Input data is from the folder 'Figure_data/', available in the associated Zenodo repository.

```
PAPER_FIGURES.ipynb  # To plot the figures from the manuscript 
PAPER_metrics.ipynb  # To calculate metrics when a neural network has been applied to a simulation
functions.py         # Some useful functions 
```

##
## 


## To create geometric masks for each simulation (note that if the geometry is not fixed then this step will need modifying)

Input files are a file with basin numbers (.nc), the domain CFG file for the cavities which are resolved in NEMO 1 degree (.nc), the meshmask for the simulation (.nc), and if necessary also the separate domain CFG file for the simulation (.nc) which contains the bathymetry and ice shelf draft files. 

Output files are the outline masks for the different grid cells (ocean, cavity, grounded ice etc), (\[simulation name\]\_geometric\_masks.nc), and the geometry variables needed for the simulation (\[simulation name\]\_geom\_vars.nc).

```
PREP_ALL_masks.ipynb # Jupyter notebook to prepare geometric masks for each simulation
```

## To go from raw NEMO files to processed files (T/S propagated for each point within the cavity) 

Input files are the raw NEMO files (.nc), plus the outline masks (geometric_masks.nc) and the geometry variables (geom_vars.nc).

Output files are the processed data for each month/year of the simulation (nn\_input\_\[simulation name\]\_y\[year\]\_m\[month\].nc). 

### For monthly data 
```
PREP_inputOPM.py      # Python script to propagate T and S
JOB_prep_for_loop.sh  # Bash script to propagate T and S 
```
### For annual data 
```
PREP_ALL_TSmelt.py   # Python script to propagate T and S for annual data 
PREP_ALL_TSmelt.sh   # Bash script to propagate T and S for annual data
```

## To merge processed files together 

Input data is geometric data for this simulation (\[simulation name\]\_geometrics\_masks.nc) and (\[simulation name\]\_geom\_vars.nc), plus the processed data for each month/year of the simulation (nn\_input\_\[simulation name\]\_y\[year\]\_m\[month\].nc)

Output data (.csv) is a processed data file with columns for each of the input variables. 
(lat, lon, temperature_prop, salinity_prop, melt_m_ice_per_y, mean_T, mean_S, std_T, std_S, year, month, basins_NEMO, distances_GL, distances_OO, distances_OC, corrected_isdraft, area, bathymetry, slope_is_lon, slope_is_lat, slope_ba_lon, slope_ba_lat, slope_is_across_front, slope_is_towards_front, slope_ba_across_front, slope_ba_towards_front) 

The processed data files for each simulation used in the manuscript are available in the associated Zenodo repository (\[simulation name\]\_whole\_dataset\_not\_yet\_normalised.csv). 

```
MERGE_OPM026.py      # Python script to merge processed files together 
MERGE_OPM026.sh      # Bash script to merge processed files together
PREP_ALL_merge.ipynb # Jupyter notebook to merge processed files together 
```

## To normalise input data for neural network training

Input data (.csv) is one or more processed data files with columns for each of the input variables. 

Output data is three files, one for the normalised metrics, one for the normalised training data, and one for the validation data (all .nc) 

```
NN_norm_input_OPM_mixed_fullruns.py  # Python script to normalise input data 
NORM_inputOPM_mixed.sh               # Bash script to normalise input data 
```

## To train a neural network 

Input data is the normalised training data and normalised validation data. 

Output data is the training neural network (.keras) and the history file for the training (.history)

The trained neural networks for each simulation used in the manuscript are available in the associated Zenodo repository (model\_nn\_small\_slope\_front\_\[simulation name\]\_\[seed]\_extrap\_std.keras). 
The final neural network trained with all available simulations is available in the associated Zenodo repository (model\_nn\_small\_slope\_front\_OPM026\_OPM0263\_OPM031\_OPM016\_OPM018\_OPM021\_ctrl94\_isf94\_isfru94\_\[seed\]\_extrap\_std.keras). 

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
