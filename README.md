# subshelf_melt_NN
This repository contains scripts which accompany the manuscript "A neural network emulator of ice-shelf melt rates for use in low resolution ocean models". 

## Paper figures 

```
PAPER_FIGURES.ipynb  # To plot the figures from the manuscript 
PAPER_metrics.ipynb  # To calculate metrics when a neural network has been applied to a simulation
functions.py         # Some useful functions 
```

## To train a neural network 

```
training.py          # Python script to train a neural network
TRAIN_seedloop.sh    # Bash script to train ensemble of 10 neural networks (seeds 1-10)
```

## To apply a trained neural network to simulation data 
```
APPLY_trainedNNtoOPM.py # Python script to apply a trained neural network to simulation data 
APPLY_trainedNNtoOPM.sh # Bash script to apply a trained neural network to simulation data 
```
