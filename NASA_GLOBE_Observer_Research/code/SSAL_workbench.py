'''
Semi-supervised Active Learning Workbench

Reads from an experiment parameter csv to run the desired experiments.
Parameter csv is found via the name passed as command-line argument

'''




from SSALDriver import SSALDriver
import argparse
import pandas as pd
import numpy as np
import os
import shutil

#parses command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("experiments_csv_name", type = str)
args = parser.parse_args()

#control whether a GPU gets used. 0 for yes, -1 for no
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#gets experiment name from based on command-line passed argument
experiments_csv_name = "./experiment_param_csvs/" + args.experiments_csv_name

#empty dataframe to store results
experiments_df = pd.read_csv(experiments_csv_name, index_col = 0 )



#loops through each experiment column
for experiment_name in experiments_df.columns:


    #create directory, save params for experiment if it doesnt already exist
    dir_name = "./SSAL_outputs/" + experiment_name + "/"
    
    is_dir = os.path.isdir(dir_name)
    
    if(is_dir == False):
    	os.mkdir(dir_name)
	
    
    #saves experiment params
    experiment_params = experiments_df[experiment_name]
    experiment_params.to_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_params.csv")
    
    #loop for each trial specificed in parameter file
    for trial_num in range(int(experiment_params.num_trials)):
    
    	#if trial has already been completed, skip it
    	#trial is completed if final predictions have been made
        trial_final_preds_path = "./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_" + str(trial_num) + "/" + experiment_name + "_" + str(trial_num) + "_final_preds.csv"
    
        final_preds_exists = os.path.isfile(trial_final_preds_path)
    
        if(final_preds_exists == True):
            print("trial already completed, skipping experiment = ", experiment_name, " trial =", trial_num)
            continue 

        print("experiment=",experiment_name, " trial=",trial_num)
        
       
        #intializes a driver to carry out the experiment
        trial_driver = SSALDriver(experiment_params, experiment_name, trial_num)
        
        #runs the experiment
        trial_driver.run_full_trial()
    
