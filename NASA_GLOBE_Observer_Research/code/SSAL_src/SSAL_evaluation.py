import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, top_k_accuracy_score, cohen_kappa_score, recall_score, precision_score
import glob
import numpy as np
import itertools
from SSAL_src.SSAL_plotting import make_pandas_bar_plot
from SSAL_src.SSAL_util import get_dataset_label_names, get_label_list
import ast


#plots experiment's training history (loss and accuracy)
def plot_experiments_history(experiment_list):

    #loops through each experiment
    for experiment_name in experiment_list:

        #get parameters
        experiment_params = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_params.csv", index_col = 0)
        num_trials = int(experiment_params.loc["num_trials"][0])


        fig = plt.figure(figsize = (24,12))


        #for trial in experiment
        for trial_num in range(num_trials):

            trial_name = experiment_name + "_" + str(trial_num)

            #get history
            trial_history = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + trial_name + "/iteration_histories/" + trial_name + "_history_0.csv", index_col = 0)


            #make loss plots
            plt.subplot(1,2,1)
            plt.plot(trial_history['loss'], linewidth = 3, label = trial_name + "_loss")
            plt.plot(trial_history['val_loss'], linewidth = 3, label = trial_name + "_val_loss")
            plt.ylim(0,3)
            plt.title(experiment_name + " loss", fontsize = 22)
            plt.xlabel("iteration", fontsize = 18)
            plt.ylabel("loss", fontsize = 18)
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.legend(loc='upper right', fontsize = "large")

            
            #make accuracy plots
            plt.subplot(1,2,2)
            plt.plot(trial_history['categorical_accuracy'], linewidth = 3, label = trial_name + "_acc")
            plt.plot(trial_history['val_categorical_accuracy'], linewidth = 3, label = trial_name + "_val_acc")
            plt.title(experiment_name + " accuracy", fontsize = 22)
            plt.xlabel("iteration", fontsize = 18)
            plt.ylabel("accuracy %", fontsize = 18 )
            plt.ylim(0.2,1.0)
            plt.tick_params(axis='both', which='major', labelsize=15)

            plt.legend(loc='upper left', fontsize = "large")




#compiles and experiment's results such as accuracy, f1, top2 acc etc. by class and overall
def compile_experiment_results(experiment_name, num_classes = 10):
    
    #empty arrays for by-class metrics
    accuracies = np.zeros(num_classes + 1)
    top2accuracies = np.zeros(num_classes + 1)
    f1s = np.zeros(num_classes + 1)

    #get experiment params
    experiment_params = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_params.csv", index_col = 0)

    num_trials = int(experiment_params.loc["num_trials"][0])

    #get column labels
    accs = get_label_list(num_classes = num_classes, base_term = "acc")
    results_columns = accs
    top2s = get_label_list(num_classes = num_classes, base_term = "top2")
    f1s_labs = get_label_list(num_classes = num_classes, base_term = "f1")

    #add all column names to same list                       
    results_columns.extend(top2s)
    results_columns.extend(f1s_labs)
   
    #results dataframe
    experiment_results = pd.DataFrame(columns = results_columns)

    #loop through each trial in experiment
    for trial_num in range(num_trials):

       
        #look at final predictions
        trial_name = experiment_name + "_" + str(trial_num)

        trial_predictions = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + trial_name + "/" + trial_name + "_final_preds.csv", index_col = 0)
        #trial_predictions["10_prob"] = 0
        

        #get detailed accuracies, top2 accuracies
        trial_true = trial_predictions["official_label"]
        trial_pred = trial_predictions["predicted_label"]
        trial_accs = get_detailed_accuracy(trial_true, trial_pred, num_classes = num_classes)

        
        
        #get detailed top2 accuracies
        trial_top2accs = get_top_k_accuracies(trial_predictions, k = 2, num_classes = num_classes)
        trial_f1s = get_detailed_f1(trial_true, trial_pred, num_classes = num_classes)
      
       
        #make data row
        data_row = np.append(trial_accs, trial_top2accs)
        data_row = np.append(data_row, trial_f1s)

        trial_results_df = pd.DataFrame(columns = results_columns, data = [data_row], index = ["trial_" + str(trial_num)])

        #append new data
        experiment_results = experiment_results.append(trial_results_df)


        #compile results from each trial for average later

        accuracies += trial_accs
        top2accuracies += trial_top2accs
        
        f1s += trial_f1s

    #average results
    avg_accuracies = accuracies/num_trials
    avg_top2accuracies = top2accuracies/num_trials
    avg_f1s = f1s / num_trials
    avg_data_row = np.append(avg_accuracies, avg_top2accuracies)
    avg_data_row = np.append(avg_data_row, avg_f1s)

    avg_results_df = pd.DataFrame(columns = results_columns, data = [avg_data_row], index = ["average"])

    experiment_results = experiment_results.append(avg_results_df)
    
    #save experiment results to a csv
    experiment_results.to_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_experiment_results.csv")
 
    
def get_experiment_avg_accuracies(experiment_name, num_classes = 10):
    
    experiment_results = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_experiment_results.csv", index_col = 0)
    
    accs = get_label_list(num_classes = num_classes, base_term = "acc")

    return experiment_results.loc["average"][accs].values


def get_experiment_avg_top2accuracies(experiment_name, num_classes = 10):
    
    experiment_results = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_experiment_results.csv", index_col = 0)
    
    top2s = get_label_list(num_classes = num_classes, base_term = "top2")
    
    return experiment_results.loc["average"][top2s].values


def get_experiment_avg_f1s(experiment_name, num_classes = 10):
    
    experiment_results = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + experiment_name + "_experiment_results.csv", index_col = 0)
    
    f1s = get_label_list(num_classes = num_classes, base_term = "f1")
    
    return experiment_results.loc["average"][f1s].values

def get_average_trunc_loss_accuracies(experiment_name):
    
    experiment_dir = "./SSAL_outputs/" + experiment_name
    experiment_params = pd.read_csv(experiment_dir + "/" + experiment_name + "_params.csv", index_col = 0)
    
    num_trials = int(experiment_params.loc["num_trials"][0])
    
    total_accuracies = np.zeros(7)
    
    for i in range(num_trials):
        
        trial_accuracy_df = pd.read_csv(experiment_dir + "/" + experiment_name + "_" + str(i) + "/" + experiment_name + "_" + str(i) + "_final_accs.csv", index_col = 0)
        
        trial_accuracies = trial_accuracy_df.iloc[0].values
        
        total_accuracies += trial_accuracies
        
    return total_accuracies/num_trials
        

def get_experiments_citizen_accuracy(experiment_name, num_classes = 10):
    
    experiment_params = pd.read_csv("./SSAL_outputs/"  + experiment_name + "/" + experiment_name + "_params.csv", index_col = 0)
    
    num_trials = int(experiment_params.loc["num_trials"][0])
    
    citizen_accuracies = np.zeros(num_classes + 1)
    
    for i in range(num_trials):
        
        #get trial's final preds
        trial_name = experiment_name + "_" + str(i)
        
        trial_final_preds = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + trial_name + "/" + trial_name + "_final_preds.csv", index_col = 0)
        
        trial_final_preds = trial_final_preds[trial_final_preds["official_label"] != 10]
        
        trial_true = trial_final_preds["official_label"]
        trial_citizen = trial_final_preds["citizen_label"]
        
        trial_cit_accs = get_detailed_accuracy(trial_true, trial_citizen, num_classes = num_classes)

        citizen_accuracies += trial_cit_accs
    
    return citizen_accuracies/num_trials


def get_experiments_citizen_f1(experiment_name, num_classes = 10):

    experiment_params = pd.read_csv("./SSAL_outputs/"  + experiment_name + "/" + experiment_name + "_params.csv", index_col = 0)
    
    
    num_trials = int(experiment_params.loc["num_trials"][0])
    
    citizen_accuracies = np.zeros(num_classes + 1)
    
    for i in range(num_trials):
        
        #get trial's final preds
        trial_name = experiment_name + "_" + str(i)
        
        trial_final_preds = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + trial_name + "/" + trial_name + "_final_preds.csv", index_col = 0)
        
        trial_final_preds = trial_final_preds[trial_final_preds["official_label"] != 10]
        
        trial_true = trial_final_preds["official_label"]
        trial_citizen = trial_final_preds["citizen_label"]
        
        trial_cit_accs = get_detailed_f1(trial_true, trial_citizen, num_classes = num_classes)

        citizen_accuracies += trial_cit_accs
    
    return citizen_accuracies/num_trials
    
def plot_citizen_accuracies(savename = "None", dataset = "GLOBE", num_classes = 11):
    
     
    all_experiments_metrics = []
    experiment_names = {}
    weird_c = []
    
    i = 0
    
    #get average citizen accuracies with same test data
    citizen_accuracies = get_experiments_citizen_accuracy("GLOBE_SSAL_mod_att1", num_classes = num_classes)
   
    all_experiments_metrics.append(citizen_accuracies[0:-1]) 
    experiment_names[i] = "citizens"
      
    X = get_dataset_label_names(dataset, num_classes = num_classes)[0:-1]
        
    title = "Citizen Accuracies"
    xlab = "Class"
    ylab = "Accuracy"
    ylim = (0,100)

    cat_list = all_experiments_metrics[0]
    
    num_cats = 1
    
    for i in range (1,num_cats):
        
        next_list = all_experiments_metrics[i]
        cat_list = np.c_[cat_list,next_list]
   
    #print(cat_list)
    cat_list = cat_list * 100.0
    df = pd.DataFrame(cat_list, index=X)
    df = df.rename(columns = experiment_names)

    #print(df)

    make_pandas_bar_plot(df, title = title, xlab = xlab, ylab = ylab, savename = savename, bar_label_fmt = "%2d", ylim = ylim)
    
    
    
    

def plot_experiment_metric_comparison(experiment_list, metric = "accuracy", savename = "None", fmt = "%2d", ylim = (0,100), num_classes = 10, dataset = "GLOBE", show_citizens = True, custom_legend_labels = None, custom_colors = None, title = "Metric Comparison", legend_loc = "best", custom_anchor = (1,1)):
    
    metric_dict = {"accuracy":get_experiment_avg_accuracies, "top2" :get_experiment_avg_top2accuracies, "f1" : get_experiment_avg_f1s}
    
    metric_func = metric_dict[metric]
    
    all_experiments_metrics = []
    experiment_names = {}
    weird_c = []
    
    i = 0
    
    #get average citizen accuracies with same test data
    all_experiment_citizen_accuracies = np.zeros(num_classes + 1)
    
    #if experiment list is empty (just want to plot citizen accuracies)
    if(len(experiment_list) == 0):
        
        print("length is 0")
        experiment_citizen_accuracies = get_experiments_citizen_accuracy("GLOBE_SSAL_mod_att1", num_classes = num_classes)
        all_experiment_citizen_accuracies += experiment_citizen_accuracies
        
        
    
    
    #for every experiment, get the detailed metric
    for experiment in experiment_list:
        
        
        #get citizen accuracy in experiment's test set, add them to total 
        if("trunc_loss" not in experiment):
            
            if(metric == "f1"):
                experiment_citizen_accuracies = get_experiments_citizen_f1(experiment, num_classes = num_classes)
            else:
                
                experiment_citizen_accuracies = get_experiments_citizen_accuracy(experiment, num_classes = num_classes)
        
            all_experiment_citizen_accuracies += experiment_citizen_accuracies
        
        if("trunc_loss" in experiment):
            
            experiment_metric_values = get_average_trunc_loss_accuracies(experiment)
            
        else:
        
            compile_experiment_results(experiment, num_classes = num_classes)
    
            experiment_metric_values = metric_func(experiment, num_classes)
        
        all_experiments_metrics.append(experiment_metric_values)
        
        experiment_names[i] = experiment
            
        i += 1
    
    #average the citizen accuracies
    if(len(experiment_list) != 0):
        final_experiment_accuracies = all_experiment_citizen_accuracies/ len(experiment_list)
    else:
        final_experiment_accuracies = all_experiment_citizen_accuracies
    if(show_citizens == True):
        
        print("should be showing citizens")
        all_experiments_metrics.append(final_experiment_accuracies)
        
        experiment_names[i] = "citizens"
    
            
    X = get_dataset_label_names(dataset, num_classes = num_classes)
        
        
       

    
    xlab = "Label"
    ylab = metric + " (%)"

    cat_list = all_experiments_metrics[0]
    
    num_cats = len(experiment_list)
    
    if(show_citizens == True):
        num_cats +=1
    
    for i in range (1,num_cats):
        
        next_list = all_experiments_metrics[i]
        cat_list = np.c_[cat_list,next_list]
   
    #print(cat_list)
    cat_list = cat_list * 100.0
    df = pd.DataFrame(cat_list, index=X)
    df = df.rename(columns = experiment_names)

    #print(df)

    make_pandas_bar_plot(df, title = title, xlab = xlab, ylab = ylab, savename = savename, bar_label_fmt = fmt, ylim = ylim, custom_legend_labels = custom_legend_labels, custom_colors = custom_colors, legend_loc = legend_loc, custom_anchor = custom_anchor)

    
    
def plot_final_train_dists(experiment_list, savename = "None", fmt = "%2d", ylim = (0,100), num_classes = 10, dataset = "GLOBE",  custom_legend_labels = None, custom_colors = None, title = "GLOBE Final Training Distributions", legend_loc = "best", custom_anchor = (1,1), yscale = "linear"):
    
    #Want to plot distribution of images in original 500 manually labeled images, distribution at end of base SSAL and at end of modified SSAL
    
    #original distributions, for each trial in each experiment add to this, average at end
    original_dists = np.zeros(num_classes)
    


    all_experiments_dists = []
    experiment_names = {}
    weird_c = []
    
    i = 0
    
       
    num_total_trials = 0
    
    #for every experiment, add original training distributions to original_dists, get average of final distributions
    for experiment in experiment_list:
        
        experiment_final_dists = np.zeros(num_classes)
        experiment_params = pd.read_csv("./SSAL_outputs/"  + experiment + "/" + experiment + "_params.csv", index_col = 0)
        num_trials = int(experiment_params.loc["num_trials"][0])
        num_iters = int(experiment_params.loc["maxIterations"][0])
        
        for j in range(num_trials):
            
            num_total_trials += 1
            
            #get original train distribution
            trial_eval_df = pd.read_csv("./SSAL_outputs/" + experiment + "/" + experiment + "_" + str(j) + "/" + experiment + "_" + str(j) + "_evaluation_df.csv", index_col = 0)
            
            trial_og_dist = ast.literal_eval(trial_eval_df.iloc[0]["training_set_composition_n"])
            trial_og_dist = trial_og_dist #/ np.sum(trial_og_dist) * 100
            
            trial_final_dist = ast.literal_eval(trial_eval_df.iloc[num_iters - 1]["training_set_composition_n"])
            trial_final_dist = trial_final_dist# / np.sum(trial_final_dist) * 100
            
            original_dists += trial_og_dist
            experiment_final_dists += trial_final_dist
            
            
            print(experiment)
            print(trial_og_dist)
            print(trial_final_dist)
         
            
        
        experiment_final_dists = experiment_final_dists / num_trials
        
        
        all_experiments_dists.append(experiment_final_dists)
        
        experiment_names[i] = experiment
            
        i += 1
    
    original_dists = original_dists / num_total_trials
    all_experiments_dists.append(original_dists)
        
    experiment_names[i] = "original distribution"
    
            
    X = get_dataset_label_names(dataset, num_classes = num_classes)[1:]
    print(X)
        
        
       

    
    xlab = "Label"
    ylab = "Number of Instances"

    cat_list = all_experiments_dists[0]
    
    num_cats = len(experiment_list) + 1
    
    
    
    for i in range (1,num_cats):
        
        next_list = all_experiments_dists[i]
        cat_list = np.c_[cat_list,next_list]
   
    
    df = pd.DataFrame(cat_list, index=X)
    df = df.rename(columns = experiment_names)

    #print(df)

    make_pandas_bar_plot(df, title = title, xlab = xlab, ylab = ylab, savename = savename, bar_label_fmt = fmt, ylim = ylim, custom_legend_labels = custom_legend_labels, custom_colors = custom_colors, legend_loc = legend_loc, custom_anchor = custom_anchor, yscale = yscale)



def fill_in_freq_gaps(labels_series, fill = 0, num_classes = 10):
    
    #print(labels_series)
    label_counts = np.full(num_classes, fill)
    
    for index, value in labels_series.items():
        
        label_counts[index] += value
        
    return label_counts


def load_labeled_data():
    
    labeled_data = pd.read_csv("./data/labeled_data.csv")
    
    return labeled_data

def load_man_cit_labeled_data():
    
    
    labeled_data = pd.read_csv("./data/labeled_data.csv", index_col = 0)
    
    labeled_data.dropna(inplace = True)
    
    return labeled_data
    
    

def get_citizen_accuracies():
    
    data = load_labeled_data()
    
    citizen_preds = data["citizen_label"]
    actual_labels = data["official_label"]
    
    accuracies = get_detailed_accuracy(actual_labels,citizen_preds)
    
    return accuracies


def load_SSAL_predictions(model_name):
    
    model_predictions = pd.read_csv("./SSAL_outputs/" + model_name + "/" + model_name + "_final_preds.csv")
    
    return model_predictions

def load_SSAL_eval_df(model_name):
    
    model_eval_df = pd.read_csv("./SSAL_outputs/" + model_name + "/" + model_name + "_evaluation_df.csv", index_col = 0)
    
    return model_eval_df
    
def plot_metric_comparison(model_list, metric = "accuracy", k = 2, savename = "None", fmt = "%2d", ylim = (0,100), num_classes = 10, dataset = "GLOBE"):
    
    metric_dict = {"accuracy":get_detailed_accuracy, "f1":get_detailed_f1, "topk" :get_top_k_accuracies, "precision" : get_detailed_precision, "recall" : get_detailed_recall}
    
    metric_func = metric_dict[metric]
    
    all_model_accuracies = []
    model_names = {}
    weird_c = []
    
    i = 0
    
    #for every model, get the detailed metric
    for model in model_list:
        
        if(model == "citizen_scientists"):
           
            labeled_data = load_man_cit_labeled_data()

            cit_predictions = labeled_data["citizen_label"]
            cit_actual_labels = labeled_data["official_label"]
            

            if(metric == "topk"):
                cit_accuracies = get_detailed_accuracy(cit_actual_labels,cit_predictions, num_classes = num_classes)
            else:
                cit_accuracies = metric_func(cit_actual_labels, cit_predictions, num_classes = num_classes)
                
            
            all_model_accuracies.append(cit_accuracies)
            
            model_names[i] = model
            
        else:
            
            model_data = load_SSAL_predictions(model)
            if(metric == "topk"):
                
                topk = get_top_k_accuracies(model, k = k, num_classes = num_classes)
                all_model_accuracies.append(topk)
                
                
            
            else:
                model_predictions_df = load_SSAL_predictions(model)

                model_predictions = model_predictions_df["predicted_label"]
                model_actual_labels = model_predictions_df["official_label"]

                model_accuracies = metric_func(model_actual_labels, model_predictions, num_classes = num_classes)

                all_model_accuracies.append(model_accuracies)

            model_names[i] = model
            
        i += 1


    title = metric + " Comparision"
    xlab = "Label"
    ylab = metric + " (%)"

    cat_list = all_model_accuracies[0]
    
    for i in range (1,len(model_list)):
        
        next_list = all_model_accuracies[i]
        cat_list = np.c_[cat_list,next_list]
   
    #print(cat_list)
    cat_list = cat_list * 100.0
    df = pd.DataFrame(cat_list, index=X)
    df = df.rename(columns = model_names)

    #print(df)

    make_pandas_bar_plot(df, title = title, xlab = xlab, ylab = ylab, savename = savename, bar_label_fmt = fmt, ylim = ylim)

    
    
def plot_natural_class_dist(savename = "None"):
    
    globe_natural_labels_df = pd.read_csv("./data/globe_dataset_labels.csv", index_col = 0)
    
    globe_natural_labels_df = globe_natural_labels_df[globe_natural_labels_df["dataset_label"] != 10]
    
    dataset_labels = globe_natural_labels_df["dataset_label"].value_counts().sort_index().values
    
    dataset_labels = dataset_labels / globe_natural_labels_df.shape[0]
   
    all_model_accuracies = []
    model_names = {}
    weird_c = []
    
    all_model_accuracies.append(dataset_labels)
    model_names[0] = "GLOBE Dataset"
    
    cat_list = all_model_accuracies[0]
    
    X = get_dataset_label_names("GLOBE", num_classes = 10)[1:]
    
  
   
    #print(cat_list)
    cat_list = cat_list * 100.0
    df = pd.DataFrame(cat_list, index=X)
    df = df.rename(columns = model_names)
    
   
    
    title = "GLOBE Class Distribution"
    xlab = "Class"
    ylab = "Percentage Proportion"
    ylim = (0,60)
    

    #print(df)

    make_pandas_bar_plot(df, title = title, xlab = xlab, ylab = ylab, savename = savename, bar_label_fmt = "%.2f", ylim = ylim)
    
    

def top_k_classified(row, k, num_classes = 10):
    
    actual_label = row["official_label"]
    
    probs = row.iloc[0:num_classes].tolist()
    
    argsorted_probs = np.argsort(probs)[-k:]
   
   
    if(actual_label in argsorted_probs):
        
       
        return 1
    else:
    
        return 0 
    

    
    
def get_top_k_accuracies(predictions, k, num_classes = 10):

    top_k_accuracies = []
    
    
   

    #dataset with topk col, 1 if topk correct, 0 if not 
    predictions["topk_correct"] = predictions.apply(top_k_classified, axis = 1, args = (k,num_classes,))
    top_k_col = predictions["topk_correct"]
    
    num_top_k_correct = top_k_col.sum()
    num_total = top_k_col.shape[0]
    top_k_accuracy = num_top_k_correct/num_total
    
    
    
    top_k_accuracies.append(top_k_accuracy)

    for i in range(num_classes):
      
        
        spec_class = predictions[predictions["official_label"] == i]
        top_k_col = spec_class["topk_correct"]
        
        num_top_k_correct = top_k_col.sum()
        num_total = top_k_col.shape[0]

        top_k_accuracy = num_top_k_correct/num_total

        top_k_accuracies.append(top_k_accuracy)
       
    return top_k_accuracies
    
def get_detailed_precision(y_actual,y_pred, average = "micro", num_classes = 10):
   
   
    total_precision = precision_score(y_actual, y_pred, average = average)
    all_precisions = [total_precision]
    
    ind_precisions = precision_score(y_actual, y_pred, average = None)
    
    all_precisions.extend(ind_precisions)
    
    return all_precisions

def get_detailed_recall(y_actual,y_pred, average = "micro", num_classes = 10):
   
   
    
    total_recall = recall_score(y_actual, y_pred, average = average)
    all_recalls = [total_recall]
    
    ind_recalls = recall_score(y_actual, y_pred, average = None)
    
    all_recalls.extend(ind_recalls)
    
    return all_recalls
    
    
  
def get_detailed_f1(y_actual,y_pred, average = "micro", num_classes = 10):
    
    
    
    total_f1 = f1_score(y_actual, y_pred, average = average)
    all_f1s = [total_f1]
    
    ind_f1s = f1_score(y_actual, y_pred, average = None)
    
    all_f1s.extend(ind_f1s)
    
    if(len(all_f1s) == num_classes):
        all_f1s.append(0)
    
    return all_f1s
    

    
    


def plot_confusion_matrix_from_file(model_name, save = False, from_frame = False, y_pred = None, y_actual = None, dataset = "GLOBE", num_classes = 10):
    
    
    base_name = "./SSAL_outputs/" + model_name + "/" + model_name
    
    if(from_frame != False):
        
        predictions = y_pred
        actual_labels = y_actual
        
    else:
    
        
    
        filename = base_name + "_final_preds.csv"
    
        predictions_df = pd.read_csv(filename)

        predictions = predictions_df["predicted_label"]
        official_labels = predictions_df["official_label"]
    
    conf_matrix  = confusion_matrix(official_labels,predictions)
    
   
    
    label_names = get_dataset_label_names(dataset, num_classes = num_classes)[1:]
  
    
    
    savename = base_name + "_conf_matrix.png"
    plot_confusion_matrix(conf_matrix, label_names, savename = savename, save = save)
    

def plot_side_by_side_confusion_matrix(model_1, model_2, title1 = "cm1", title2 = "cm2", cmap = None, normalize = True, savename = None, save = True, dataset = "GLOBE", num_classes = 10):
    
    
    if(model_1 == "citizen_scientists"):
        
        labeled_data = load_labeled_data()

        cit_predictions = labeled_data["citizen_label"]
        cit_actual_labels = labeled_data["official_label"]
        
        cm1 = confusion_matrix(cit_actual_labels, cit_predictions)
        
    else:
        
        model_predictions_df = load_SSAL_predictions(model_1)

        model_predictions = model_predictions_df["predicted_label"]
        model_actual_labels = model_predictions_df["official_label"]
        
        cm1 = confusion_matrix(model_actual_labels, model_predictions)
        
    if(model_2 == "citizen_scientists"):
        
        labeled_data = load_labeled_data()

        cit_predictions = labeled_data["citizen_label"]
        cit_actual_labels = labeled_data["official_label"]
        
        cm2 = confusion_matrix(cit_actual_labels, cit_predictions)
        
    else:
        
        model_predictions_df = load_SSAL_predictions(model_2)

        model_predictions = model_predictions_df["predicted_label"]
        model_actual_labels = model_predictions_df["official_label"]
        
        cm2 = confusion_matrix(model_actual_labels, model_predictions)
        
     
   
    
    target_names = get_dataset_label_names(dataset, num_classes = num_classes)[1:]
    
    
    fig = plt.figure(figsize = (18,12))
    
    plt.subplot(1,2,1)
    accuracy = np.trace(cm1) / np.sum(cm1).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    #plt.figure(figsize=(16, 12))
    plt.imshow(cm1, interpolation='nearest', cmap=cmap)
    plt.title(model_1)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]


    thresh = cm1.max() / 1.50 if normalize else cm1.max() / 2.0
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm1[i, j]),
                     horizontalalignment="center",
                     color="white" if cm1[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm1[i, j]),
                     horizontalalignment="center",
                     color="white" if cm1[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    
    plt.subplot(1,2,2)
    accuracy = np.trace(cm2) / np.sum(cm2).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    #plt.figure(figsize=(16, 12))
    plt.imshow(cm2, interpolation='nearest', cmap=cmap)
    plt.title(model_2)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks([])

    if normalize:
        cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]


    thresh = cm2.max() / 1.5 if normalize else cm2.max() / 2
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm2[i, j]),
                     horizontalalignment="center",
                     color="white" if cm2[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm2[i, j]),
                     horizontalalignment="center",
                     color="white" if cm2[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
def plot_citizen_confusion_matrix(savename = None, save = False, dataset = "GLOBE", num_classes = 10, title = None):
    
    experiment_name = "GLOBE_SSAL_mod_att1"
    trial_name = experiment_name + "_" + str(0)
    
    sample_final_preds = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + trial_name + "/" + trial_name + "_final_preds.csv", index_col = 0)
    
    sample_final_preds = sample_final_preds[sample_final_preds["official_label"] != 10]
    
    citizen_predictions = sample_final_preds["citizen_label"].values
    true_labels = sample_final_preds["official_label"].values
    
    cm = confusion_matrix(true_labels, citizen_predictions)
    
    target_names = get_dataset_label_names(dataset = dataset, num_classes = num_classes)[1:]
    
 
    
    plot_confusion_matrix(cm, target_names, title = "Citizen Label Confusion Matrix", savename = savename, save = save, show_bar = False)

    
def plot_experiment_confusion_matrix(experiment_name, dataset = "GLOBE", num_classes = 11, title = None, savename = None, save = False):
 
    true_labels = []
    predictions = []
    
    
    experiment_params = pd.read_csv("./SSAL_outputs/"  + experiment_name + "/" + experiment_name + "_params.csv", index_col = 0)
    
    
    num_trials = int(experiment_params.loc["num_trials"][0])
    
    for i in range(num_trials):
        
        #get trial's final preds
        trial_name = experiment_name + "_" + str(i)
        
        trial_final_preds = pd.read_csv("./SSAL_outputs/" + experiment_name + "/" + trial_name + "/" + trial_name + "_final_preds.csv", index_col = 0)
        
        trial_true = trial_final_preds["official_label"]
        trial_predictions = trial_final_preds["predicted_label"]
        
        true_labels.extend(trial_true)
        predictions.extend(trial_predictions)

    cm = confusion_matrix(true_labels, predictions)
    target_names = get_dataset_label_names(dataset = dataset, num_classes = num_classes)[1:]
    
    plot_confusion_matrix(cm, target_names,title = title, savename = savename, save = save, show_bar = False)

def plot_confusion_matrix(cm,
                          target_names,
                          title=None,
                          cmap=None,
                          normalize=True, savename = None, save = False, show_bar = True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    if(title is None):
        plt.title("Conf Matrix", fontsize = 20)
    else:
        plt.title(title, fontsize = 20)
    
    if(show_bar == True):
        plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize = 10)
        plt.yticks(tick_marks, target_names, fontsize = 10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    
    plt.ylabel('True label', fontsize = 16)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize = 16)
    #plt.show()
    plt.tight_layout()
    if(save == True):
        plt.savefig(savename)
    
    


def get_detailed_accuracy(y_true,y_pred, num_classes = 10):
    
   
    
    total_accuracy = accuracy_score(y_true,y_pred)
    
    all_accuracies = [total_accuracy]
    
    conf_matrix = confusion_matrix(y_true,y_pred)
    
    byclass_accuracies = (conf_matrix.diagonal()/conf_matrix.sum(axis=1)).tolist()
    
    all_accuracies.extend(byclass_accuracies)
    
    if(len(all_accuracies) == num_classes):
        all_accuracies.append(0)
    
    return all_accuracies



def random_label_func(row, num_classes = 10):
    
    return np.random.randint(num_classes)

def boost_changed_confident_func(row):
    
    if(row["pre_boosted_confident"] == row["post_boosted_confident"]):
        
        return 0
    
    else:
        
        return 1 
    
def post_boost_corr_label_func(row):
    
    if(row["post_boosted_prediction"] == row["dataset_label"]):
        
        return 1
    
    else:
        
        return 0
    
def post_boost_corr_label_func_globe(row):
    
    if(int(row["post_boosted_prediction"]) == int(row["official_label"])):
        
        return 1
    
    else:
        
        return 0
    
def pre_boost_corr_label_func(row):
    
    if(row["pre_boosted_prediction"] == row["dataset_label"]):
        
        return 1
    
    else:
        
        return 0   
    
def pre_boost_corr_label_func_globe(row):
    
    if(int(row["pre_boosted_prediction"]) == int(row["official_label"])):
        
        return 1
    
    else:
        
        return 0  

    
    
def get_prediction_stats(model_name):
    iteration_predictions_df = pd.DataFrame()

    pseudolabel_dir = "./SSAL_outputs/" + model_name + "/iteration_predictions/"



    iteration_files = glob.glob(pseudolabel_dir + "*predictions*")

    for file in iteration_files:

        iter_prediction = pd.read_csv(file) 

        iteration_predictions_df = iteration_predictions_df.append(iter_prediction)

    #iteration_predictions_df["official_label"] = iteration_predictions_df.apply(random_label_func, axis = 1)
    iteration_predictions_df["boost_changed_confident"] = iteration_predictions_df.apply(boost_changed_confident_func, axis = 1)
    iteration_predictions_df["pre_boost_correct_pseudolabel"] = iteration_predictions_df.apply(pre_boost_corr_label_func, axis = 1)
    iteration_predictions_df["post_boost_correct_pseudolabel"] = iteration_predictions_df.apply(post_boost_corr_label_func, axis = 1)
    
    total_predictions = iteration_predictions_df.shape[0]

    boost_changed_confident = iteration_predictions_df[iteration_predictions_df["boost_changed_confident"] == 1]
    pre_boost_confident = iteration_predictions_df[iteration_predictions_df["pre_boosted_confident"] == 1]
    post_boost_confident = iteration_predictions_df[iteration_predictions_df["post_boosted_confident"] == 1]

    num_boost_changed_confident = boost_changed_confident.shape[0]
    total_pre_boost_confident = pre_boost_confident.shape[0]
    total_post_boost_confident = post_boost_confident.shape[0]
    per_conf_from_boost = num_boost_changed_confident/total_post_boost_confident * 100

    pre_boost_accuracy = pre_boost_confident["pre_boost_correct_pseudolabel"].sum()/total_pre_boost_confident*100
    post_boost_accuracy = post_boost_confident["post_boost_correct_pseudolabel"].sum()/total_post_boost_confident*100
    boost_changed_confidence_accuracy = boost_changed_confident["post_boost_correct_pseudolabel"].sum()/num_boost_changed_confident*100
    
    #number of predictions that became confident with boost
    print("number of predictions that became confident with boost", num_boost_changed_confident)
    #percentage of pseudolabels that are confident because of boost
    print("percentage of confident predictions that are confident because of boost", per_conf_from_boost)
    #accuracy among pseudolabels before boosting
    print("accuracy among confident predictions before boosting", pre_boost_accuracy)
    #accuracy among pseudolabels after boosting
    print("accuracy among confident predictions after boosting", post_boost_accuracy)
    #accuracy among pseudolabels which switched confident
    print("accuracy among confident predictions which switched confident", boost_changed_confidence_accuracy)
    
    #stats for pseudolabels
    total_pseudolabels = iteration_predictions_df[iteration_predictions_df["used_as_pseudolabel"] == 1].shape[0]
    boost_changed_confident_pl = boost_changed_confident[boost_changed_confident["used_as_pseudolabel"] == 1]
    pre_boost_confident_pl = pre_boost_confident[pre_boost_confident["used_as_pseudolabel"] == 1]
    post_boost_confident_pl = post_boost_confident[post_boost_confident["used_as_pseudolabel"] == 1]

    num_boost_changed_confident_pl = boost_changed_confident_pl.shape[0]
    total_pre_boost_confident_pl = pre_boost_confident_pl.shape[0]
    total_post_boost_confident_pl = post_boost_confident_pl.shape[0]
    per_conf_from_boost_pl = num_boost_changed_confident_pl/total_post_boost_confident_pl * 100

    pre_boost_accuracy_pl = pre_boost_confident_pl["pre_boost_correct_pseudolabel"].sum()/total_pre_boost_confident_pl*100
    post_boost_accuracy_pl = post_boost_confident_pl["post_boost_correct_pseudolabel"].sum()/total_post_boost_confident_pl*100
    boost_changed_confidence_accuracy_pl = boost_changed_confident_pl["post_boost_correct_pseudolabel"].sum()/num_boost_changed_confident_pl*100
    
    print()
    print()
    #number of predictions that became confident with boost
    print("number of pseudolabels that became confident with boost", num_boost_changed_confident_pl)
    #percentage of pseudolabels that are confident because of boost
    print("percentage of pseudolabels that are confident because of boost", per_conf_from_boost_pl)
    #accuracy among pseudolabels before boosting
    print("accuracy among pseudolabels before boosting", pre_boost_accuracy_pl)
    #accuracy among pseudolabels after boosting
    print("accuracy among pseudolabels after boosting", post_boost_accuracy_pl)
    #accuracy among pseudolabels which switched confident
    print("accuracy among pseudolabels which switched confident", boost_changed_confidence_accuracy_pl)
    
    
    
    


    savename = pseudolabel_dir + model_name + "_pseudolabel_results.csv"
    print("saving results to ",savename)

    columns = ["total_predictions","num_pre_boost_confident","num_post_boost_confident","num_changed_confident","per_conf_from_boost","pre_boost_accuracy","post_boost_accuracy","boost_changed_confidence_accuracy","total_pseudolabels","num_pre_boost_confident_pl","num_post_boost_confident_pl","num_changed_confident_pl","per_conf_from_boost_pl","pre_boost_accuracy_pl","post_boost_accuracy_pl","boost_changed_confidence_accuracy_pl"]
    data = [total_predictions, total_pre_boost_confident, total_post_boost_confident, num_boost_changed_confident, per_conf_from_boost, pre_boost_accuracy, post_boost_accuracy, boost_changed_confidence_accuracy, total_pseudolabels, total_pre_boost_confident_pl, total_post_boost_confident_pl, num_boost_changed_confident_pl, per_conf_from_boost_pl, pre_boost_accuracy_pl, post_boost_accuracy_pl, boost_changed_confidence_accuracy_pl]

    results_df = pd.DataFrame(data = [data], columns = columns)

    results_df.to_csv(savename)    
    
    
    
def get_pseudolabel_stats(model_name):
    iteration_predictions_df = pd.DataFrame()

    pseudolabel_dir = "./SSAL_outputs/" + model_name + "/iteration_predictions/"



    iteration_files = glob.glob(pseudolabel_dir + "*predictions*")
    print(iteration_files)
    
    for file in iteration_files:

        iter_prediction = pd.read_csv(file) 

        iteration_predictions_df = iteration_predictions_df.append(iter_prediction)

    print(iteration_predictions_df.shape[0])
    
    pseudolabels = iteration_predictions_df[iteration_predictions_df["used_as_pseudolabel"] == 1]
    
    print(pseudolabels.shape[0])
        
    iteration_predictions_df["boost_changed_confident"] = iteration_predictions_df.apply(boost_changed_confident_func, axis = 1)
    iteration_predictions_df["pre_boost_correct_pseudolabel"] = iteration_predictions_df.apply(pre_boost_corr_label_func, axis = 1)
    iteration_predictions_df["post_boost_correct_pseudolabel"] = iteration_predictions_df.apply(post_boost_corr_label_func, axis = 1)
    
    boost_changed_confident = iteration_predictions_df[iteration_predictions_df["boost_changed_confident"] == 1]
    pre_boost_confident = iteration_predictions_df[iteration_predictions_df["pre_boosted_confident"] == 1]
    post_boost_confident = iteration_predictions_df[iteration_predictions_df["post_boosted_confident"] == 1]
    
    #stats for pseudolabels
    total_pseudolabels = iteration_predictions_df[iteration_predictions_df["used_as_pseudolabel"] == 1].shape[0]
    boost_changed_confident_pl = boost_changed_confident[boost_changed_confident["used_as_pseudolabel"] == 1]
    pre_boost_confident_pl = pre_boost_confident[pre_boost_confident["used_as_pseudolabel"] == 1]
    post_boost_confident_pl = post_boost_confident[post_boost_confident["used_as_pseudolabel"] == 1]

    num_boost_changed_confident_pl = boost_changed_confident_pl.shape[0]
    total_pre_boost_confident_pl = pre_boost_confident_pl.shape[0]
    total_post_boost_confident_pl = post_boost_confident_pl.shape[0]
    per_conf_from_boost_pl = num_boost_changed_confident_pl/total_post_boost_confident_pl * 100

    pre_boost_accuracy_pl = pre_boost_confident_pl["pre_boost_correct_pseudolabel"].sum()/total_pre_boost_confident_pl*100
    post_boost_accuracy_pl = post_boost_confident_pl["post_boost_correct_pseudolabel"].sum()/total_post_boost_confident_pl*100
    boost_changed_confidence_accuracy_pl = boost_changed_confident_pl["post_boost_correct_pseudolabel"].sum()/num_boost_changed_confident_pl*100
    
    print()
    print()
    print("total number of pseudolabels", total_pseudolabels)
    #number of predictions that became confident with boost
    print("number of pseudolabels that became confident with boost", num_boost_changed_confident_pl)
    #percentage of pseudolabels that are confident because of boost
    print("percentage of pseudolabels that are confident because of boost", per_conf_from_boost_pl)
    #accuracy among pseudolabels before boosting
    print("accuracy among pseudolabels before boosting", pre_boost_accuracy_pl)
    #accuracy among pseudolabels after boosting
    print("accuracy among pseudolabels after boosting", post_boost_accuracy_pl)
    #accuracy among pseudolabels which switched confident
    print("accuracy among pseudolabels which switched confident", boost_changed_confidence_accuracy_pl)
    
    
    
    


    savename = pseudolabel_dir + model_name + "_pseudolabel_results.csv"
    print("saving results to ",savename)

    columns = ["num_pre_boost_confident_pl","num_post_boost_confident_pl","num_changed_confident_pl","per_conf_from_boost_pl","pre_boost_accuracy_pl","post_boost_accuracy_pl","boost_changed_confidence_accuracy_pl"]
    data = [total_pre_boost_confident_pl, total_post_boost_confident_pl, num_boost_changed_confident_pl, per_conf_from_boost_pl, pre_boost_accuracy_pl, post_boost_accuracy_pl, boost_changed_confidence_accuracy_pl]

    results_df = pd.DataFrame(data = [data], columns = columns)

    results_df.to_csv(savename)    
    
    
def get_prediction_stats_experiment(experiment, limit_to_labeled = False, dataset = "GLOBE"):
    iteration_predictions_df = pd.DataFrame()

    
    experiment_params = pd.read_csv("./SSAL_outputs/" + experiment + "/" + experiment + "_params.csv", index_col = 0)
    num_trials = int(experiment_params[experiment].num_trials)
    
    #for each trial, add each iteration's predictions to itertion_predictions_df
    
    for n in range(num_trials):
    
        trial_name = experiment + "_" + str(n)
        pseudolabel_dir = "./SSAL_outputs/" + experiment + "/" + trial_name + "/" + "iteration_predictions/"



        iteration_files = glob.glob(pseudolabel_dir + "*predictions*")

        for file in iteration_files:

            iter_prediction = pd.read_csv(file) 

            iteration_predictions_df = iteration_predictions_df.append(iter_prediction)

    #iteration_predictions_df["official_label"] = iteration_predictions_df.apply(random_label_func, axis = 1)
    
    #only get rows where we have already assigned 
    
    if(limit_to_labeled == True):
        
        print(iteration_predictions_df.shape[0])
        iteration_predictions_df = iteration_predictions_df.loc[iteration_predictions_df.official_label.notnull()]
        print(iteration_predictions_df.shape[0])
        
    
    iteration_predictions_df["boost_changed_confident"] = iteration_predictions_df.apply(boost_changed_confident_func, axis = 1)
    try:
        iteration_predictions_df["pre_boost_correct_pseudolabel"] = iteration_predictions_df.apply(pre_boost_corr_label_func_globe, axis = 1)
    except:
        iteration_predictions_df["pre_boost_correct_pseudolabel"] = iteration_predictions_df.apply(pre_boost_corr_label_func, axis = 1)
        
    try:
        iteration_predictions_df["post_boost_correct_pseudolabel"] = iteration_predictions_df.apply(post_boost_corr_label_func_globe, axis = 1)
        
    except:
        iteration_predictions_df["post_boost_correct_pseudolabel"] = iteration_predictions_df.apply(post_boost_corr_label_func, axis = 1)
    
    total_predictions = iteration_predictions_df.shape[0]

    boost_changed_confident = iteration_predictions_df[iteration_predictions_df["boost_changed_confident"] == 1]
    pre_boost_confident = iteration_predictions_df[iteration_predictions_df["pre_boosted_confident"] == 1]
    post_boost_confident = iteration_predictions_df[iteration_predictions_df["post_boosted_confident"] == 1]

    num_boost_changed_confident = boost_changed_confident.shape[0]
    total_pre_boost_confident = pre_boost_confident.shape[0]
    total_post_boost_confident = post_boost_confident.shape[0]
    per_conf_from_boost = num_boost_changed_confident/total_post_boost_confident * 100

    pre_boost_accuracy = pre_boost_confident["pre_boost_correct_pseudolabel"].sum()/total_pre_boost_confident*100
    post_boost_accuracy = post_boost_confident["post_boost_correct_pseudolabel"].sum()/total_post_boost_confident*100
    boost_changed_confidence_accuracy = boost_changed_confident["post_boost_correct_pseudolabel"].sum()/num_boost_changed_confident*100
    
    #number of predictions that became confident with boost
    
    print("total number of predictions", iteration_predictions_df.shape[0]/num_trials)
    print("percentage of predictions that are pre-boost confident", total_pre_boost_confident/total_predictions)
    print("percentage of predictiosn that are post-boost confident", total_post_boost_confident/total_predictions)
    print("total number of post-boost confident predictions", post_boost_confident.shape[0]/num_trials)
    print("number of predictions that became confident with boost", num_boost_changed_confident)
    #percentage of pseudolabels that are confident because of boost
    print("percentage of confident predictions that are confident because of boost", per_conf_from_boost)
    #accuracy among pseudolabels before boosting
    print("accuracy among confident predictions before boosting", pre_boost_accuracy)
    #accuracy among pseudolabels after boosting
    print("accuracy among confident predictions after boosting", post_boost_accuracy)
    #accuracy among pseudolabels which switched confident
    print("accuracy among confident predictions which switched confident", boost_changed_confidence_accuracy)
    
    #stats for pseudolabels
    total_pseudolabels = iteration_predictions_df[iteration_predictions_df["used_as_pseudolabel"] == 1].shape[0]
    boost_changed_confident_pl = boost_changed_confident[boost_changed_confident["used_as_pseudolabel"] == 1]
    pre_boost_confident_pl = pre_boost_confident[pre_boost_confident["used_as_pseudolabel"] == 1]
    post_boost_confident_pl = post_boost_confident[post_boost_confident["used_as_pseudolabel"] == 1]

    num_boost_changed_confident_pl = boost_changed_confident_pl.shape[0]
    total_pre_boost_confident_pl = pre_boost_confident_pl.shape[0]
    total_post_boost_confident_pl = post_boost_confident_pl.shape[0]
    per_conf_from_boost_pl = num_boost_changed_confident_pl/total_post_boost_confident_pl * 100

    pre_boost_accuracy_pl = pre_boost_confident_pl["pre_boost_correct_pseudolabel"].sum()/total_pre_boost_confident_pl*100
    post_boost_accuracy_pl = post_boost_confident_pl["post_boost_correct_pseudolabel"].sum()/total_post_boost_confident_pl*100
    boost_changed_confidence_accuracy_pl = boost_changed_confident_pl["post_boost_correct_pseudolabel"].sum()/num_boost_changed_confident_pl*100
    
    print()
    print()
    print("total number of pseudolabels", total_pseudolabels/num_trials)
    #number of predictions that became confident with boost
    print("number of pseudolabels that became confident with boost", num_boost_changed_confident_pl/num_trials)
    #percentage of pseudolabels that are confident because of boost
    print("percentage of pseudolabels that are confident because of boost", per_conf_from_boost_pl)
    #accuracy among pseudolabels before boosting
    print("accuracy among pseudolabels before boosting", pre_boost_accuracy_pl)
    #accuracy among pseudolabels after boosting
    print("accuracy among pseudolabels after boosting", post_boost_accuracy_pl)
    #accuracy among pseudolabels which switched confident
    print("accuracy among pseudolabels which switched confident", boost_changed_confidence_accuracy_pl)
    
    
    
    


    savename = "./SSAL_outputs/" + experiment + "/" + "prediction_results.csv"
    print("saving results to ",savename)

    columns = ["total_predictions","num_pre_boost_confident","num_post_boost_confident","num_changed_confident","per_conf_from_boost","pre_boost_accuracy","post_boost_accuracy","boost_changed_confidence_accuracy","total_pseudolabels","num_pre_boost_confident_pl","num_post_boost_confident_pl","num_changed_confident_pl","per_conf_from_boost_pl","pre_boost_accuracy_pl","post_boost_accuracy_pl","boost_changed_confidence_accuracy_pl"]
    data = [total_predictions, total_pre_boost_confident, total_post_boost_confident, num_boost_changed_confident, per_conf_from_boost, pre_boost_accuracy, post_boost_accuracy, boost_changed_confidence_accuracy, total_pseudolabels, total_pre_boost_confident_pl, total_post_boost_confident_pl, num_boost_changed_confident_pl, per_conf_from_boost_pl, pre_boost_accuracy_pl, post_boost_accuracy_pl, boost_changed_confidence_accuracy_pl]

    results_df = pd.DataFrame(data = [data], columns = columns)

    results_df.to_csv(savename)        
    
