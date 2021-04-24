# Cherrypick-synthetic

## Introduction

This is a repository containing code and sample data for the project *Cherrypick* . 

## Pre-requisites


This code depends on the following packages:

* numpy

* matplotlib

* scipy

* sklearn

## Code structure 
. 


- Code  : codes

- Code/run_code.py : code for simulating experiments 

- Code/cherrypick\_x.py : class definitions proposed method 

- Data : Sample Datasets

## Execution
The following command 
> python3 run_code.py -p path_to_file -t forecasting_time -m subset_selection_method  -f fraction_of_data_to_be_selected -l lambda


Will execute specific subset selection followed by opinion forecasting on test set of given dataset and will print mean squared error of the prediction with actual opinion values.

For example, 
> python3 run_code.py -p ../Data/bar_al_n_0.5 -t 0.1 -m cherrypick_a -f 0.8 -l 1.0



Path_to_file:
> *graph_n_x*
where x can take value in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
 and graph can take value in [bar_al, cp512_k, k_512]

Valid methods : 
> *cherrypick\_a* \
>  *cherrypick\_d* \
> *cherrypick\_e* \
> *cherrypick\_t* 
