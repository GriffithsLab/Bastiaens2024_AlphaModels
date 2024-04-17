# Overview

This repository contains the code and results related to the paper "A comprehensive investigation of intracortical and corticothalamic models of alpha rhythms" by SP Bastiaens, D Momi and JD Griffiths (https://doi.org/10.1101/2024.03.01.583035) which focuses on the Jansen-Rit (JR), Moran-David-Friston (MDF), Liley-Wright (LW), and Robinson-Rennie-Wright (RRW) models in the alpha regime.

## Folders
- ```code```: Contains the .py files to run the simulations to reproduce the figures from the paper.
- ```notebooks```: Contains the .ipynb files to run the simulations to reproduce the figures from the paper.
- ```data```: Contains the simulation results of the connectivity heatmaps as the run time of the simulations are longer. 

## Files
- ```Generate_time_series_power_spectra```: Numerical simulations of the four studied models (JR, MDF, LW, and RRW) and the power spectra with parameters to generate alpha oscillations (Fig. 5, 6, 7, and 8).
- ```Generate_analytical_power_spectra```: Analytical simulations of the four studied models with the analytical power spectra (Fig. 5, 6, 7, and 8).
- ```Eyes_closed_eyes_open```: Simulations with parameters to generate eyes closed and eyes open results (Fig. 9).
- ```Rate_Constant_Heatmap```: Determines the alpha peak frequencies for different parameter values of the excitatory and inhibitory time constants for the JR, MDF, and LW models (Fig. 10).
- ```Connectivity_Heatmaps```: Determines the alpha peak frequencies for different parameter values of the E-I connectivity constants for JR, LW, and RRW (Fig. 11).
- ```JR_fixed_points```: Stability analysis of the JR model with high and low noise for sets of connectivity constants (Fig. 12).
- ```LW_fixed_points```: Stability analysis of the LW model with high and low noise for sets of connectivity constants (Fig. 12).
- ```Sigmoid_and_Impulse_Response```: Sigmoid and impulse response curves for all the models for different values of firing thresholds and rate constants respectively (Fig. 13, and 14).
- ```Estimate_1_f_component```: Methods on how the 1/f pre and post peak slope were calculated.

## Usage
- Install libraries: ```pip install -r requirements.txt```
- Run python script example ```python code/Generate_time_series_power_spectra.py```
- Run jupyter notebook files: Open a jupyter notebook and run the .ipynb file of interest
