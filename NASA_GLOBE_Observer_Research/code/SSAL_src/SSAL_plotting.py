import matplotlib.pyplot as plt
import pandas as pd

import ast
import sys
from SSAL_src.SSAL_evaluation import *
from SSAL_src.SSAL_util import *

def load_SSAL_eval_df(model_name):
    
    model_eval_df = pd.read_csv("./SSAL_outputs/" + model_name + "/" + model_name + "_evaluation_df.csv", index_col = 0)
    
    return model_eval_df


def load_labeled_data():
    
    labeled_data = pd.read_csv("./data/labeled_data.csv")
    
    return labeled_data

def get_citizen_accuracies():
    
    data = load_labeled_data()
    
    citizen_preds = data["citizen_label"]
    actual_labels = data["official_label"]
    
    accuracies = get_detailed_accuracy(actual_labels,citizen_preds)
    
    return accuracies

def make_bar_plot(X,Y, color = "#18453B", xlab = "xlabel", ylab = "ylabel", title = "title", savename = "None", freq_text_center = 0.03, custom_x_labels = None):
    
    fig, ax = plt.subplots(figsize = (16,8))


    plt.bar(X,Y, linewidth = 4,edgecolor = "white")

    plt.xlabel(xlab,fontsize = 20, labelpad = 20)
    plt.ylabel(ylab, fontsize = 20, labelpad = 20)
    plt.title(title, fontsize = 26)

    if(custom_x_labels != None):
       
        y_pos = np.arange(len(custom_x_labels))
        plt.xticks(y_pos, custom_x_labels)


    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    plt.tick_params(length = 6, width = 3)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    #for i, v in enumerate(Y):
       # ax.text(i - freq_text_center * len(str(v)), 800, str(v), color='black', fontweight='bold',fontsize = 18)
    for container in ax.containers:
        
        print(container)
        ax.bar_label(container, padding = 1, fmt = "%.2f", fontsize = 8)
        
    plt.tight_layout()
    
    if(savename != "None"):
        
        plt.savefig(savename, dpi = 100)
        
        
def make_pandas_bar_plot(df, colors ="", xlab = "xlabel", ylab = "ylabel", title = "title", savename = "None", bar_label_fmt = "%.2f", edgecolor = "white", primarycolor = "#18453B", ylim = "None", stacked = False, label_type = "edge", custom_legend_labels = None, custom_colors = None, legend_loc = "best", custom_anchor = (1,1), yscale = "linear"):
    
    if(custom_colors is None):
        ax = df.plot.bar(figsize = (14,8), linewidth = 2,edgecolor = "white", stacked = stacked)#, color = primarycolor)
    else:
        ax = df.plot.bar(figsize = (14,8), linewidth = 3,edgecolor = "white", stacked = stacked, color = custom_colors)
 
    plt.xlabel(xlab,fontsize = 20, labelpad = 20)
    plt.ylabel(ylab, fontsize = 20, labelpad = 10)
    plt.title(title, fontsize = 26)

    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    if(ylim != "None"):
        plt.ylim(ylim)

    plt.tick_params(length = 6, width = 3)

    plt.yscale(yscale)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    #for i, v in enumerate(Y):
       # ax.text(i - freq_text_center * len(str(v)), 800, str(v), color='black', fontweight='bold',fontsize = 18)

    for container in ax.containers:
        
  
        ax.bar_label(container, padding = 0, fmt = bar_label_fmt, fontsize = 8, label_type = label_type)
    
      
    if(custom_legend_labels is None):    
        plt.legend(fontsize = 12, bbox_to_anchor = custom_anchor, loc = legend_loc)
    else:
        plt.legend(fontsize = 12, bbox_to_anchor = custom_anchor, labels = custom_legend_labels, loc = legend_loc)
    plt.tight_layout()
    
    if(savename != "None"):
        
        plt.savefig(savename, dpi = 400)
        
def make_training_set_comp_plot(model_name, savename = None):

    path_name = "./SSAL_evaluations/" + model_name + "/"
    file_name = path_name + model_name + "_evaluation_df.csv"

    df = pd.read_csv(file_name, index_col = 0)
    df = df[["iteration","num_manual_labeled","num_web","num_model_labeled","num_hackathon_labeled"]]    

    ax = df.plot.area(x = "iteration",figsize = (14,8))

    xlab = "iteration"
    ylab = "number of instances"
    title = "composition of training set vs iteration - " + model_name    

    plt.xlabel(xlab,fontsize = 20, labelpad = 20)
    plt.ylabel(ylab, fontsize = 20, labelpad = 10)
    plt.title(title, fontsize = 26)

    plt.xticks(range(min(df["iteration"]),max(df["iteration"]) + 1),fontsize = 15)
    plt.yticks(fontsize = 15)



    plt.tick_params(length = 6, width = 3)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)


    plt.legend(fontsize = 10, bbox_to_anchor = (1,1))
    plt.tight_layout()
    
    if(savename is not None):
        plt.savefig(path_name + savename)
    else:
        plt.savefig(path_name + model_name + "_training_area_plot.png", dpi = 400)
def make_training_set_comp_plot_experiment(experiment, savename = None, title = None, ylim = None,):

    
    experiment_params = pd.read_csv("./SSAL_outputs/" + experiment + "/" + experiment + "_params.csv", index_col = 0)
    num_trials = int(experiment_params[experiment].num_trials)
    num_iterations = int(experiment_params[experiment].maxIterations)
    
    iterations = np.arange(num_iterations)
    manual_labeled = np.zeros(num_iterations)
    model_labeled = np.zeros(num_iterations)
    hackathon_labeled = np.zeros(num_iterations)
    #want the average composition at each trial
    
    #for each trial
    for i in range(num_trials):
        trial_name = experiment + "_" + str(i) 
        trial_evaluation_df = pd.read_csv("./SSAL_outputs/" + experiment + "/" + trial_name + "/" + trial_name + "_evaluation_df.csv", index_col = 0)
        for n in range(num_iterations):
         
            #add accuracy at each iteration
            
            
            manual_labeled[n] += trial_evaluation_df.iloc[n]["num_manual_labeled"]
            model_labeled[n] += trial_evaluation_df.iloc[n]["num_model_labeled"]
            hackathon_labeled[n] += trial_evaluation_df.iloc[n]["num_hackathon_labeled"]
    
    manual_labeled = manual_labeled/num_trials
    model_labeled = model_labeled/num_trials
    hackathon_labeled = hackathon_labeled/num_trials
    
    df = pd.DataFrame()
    
    df["iteration"] = iterations
    df["num_manual_labeled"] = manual_labeled
    df["num_model_labeled"] = model_labeled
    df["num_hackathon_labeled"] = hackathon_labeled
  

    ax = df.plot.area(x = "iteration",figsize = (10,6), colormap = "Accent")

    xlab = "iteration"
    ylab = "number of instances"
    
    if(ylim != None):
        plt.ylim(ylim)
    
    if(title == None):
        title = "composition of training set vs iteration - " + experiment    

    plt.xlabel(xlab,fontsize = 20, labelpad = 20)
    plt.ylabel(ylab, fontsize = 20, labelpad = 10)
    plt.title(title, fontsize = 24, pad = 25)

    plt.xticks(range(min(df["iteration"]),max(df["iteration"]) + 1),fontsize = 15)
    plt.yticks(fontsize = 15)



    plt.tick_params(length = 6, width = 3)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)


    plt.legend(loc = "upper left",fontsize = 14, labels = ["og manual labeled", "pseudolabels", "active learning labeled"])
    plt.tight_layout()
    
    if(savename is not None):
        plt.savefig(savename, dpi = 400, bbox_inches = "tight")
    else:
        plt.savefig("./SSAL_outputs/" + "tset_plot", dpi = 400, bbox_inches = "tight") 
        
def make_al_experiment_plot(experiment_names, savename = None, custom_legend_labels = None):
    
  
    #each experiment has x number of trials, n number of iterations
    
    fig,ax = plt.subplots(figsize = (12,8))
    
    #for each experiment
    for experiment in experiment_names:
        
        
        
        
        experiment_params = pd.read_csv("./SSAL_outputs/" + experiment + "/" + experiment + "_params.csv", index_col = 0)
        num_trials = int(experiment_params[experiment].num_trials)
        num_iterations = int(experiment_params[experiment].maxIterations)
        
        accuracies = np.zeros(num_iterations)
        num_hackathon_manual_images = np.zeros(num_iterations)
        
        
        #for each trial
        for i in range(num_trials):
           
            
            trial_name = experiment + "_" + str(i) 
            trial_evaluation_df = pd.read_csv("./SSAL_outputs/" + experiment + "/" + trial_name + "/" + trial_name + "_evaluation_df.csv", index_col = 0)
            for n in range(num_iterations):
                
                #add accuracy at each iteration
                accuracies[n] += trial_evaluation_df.iloc[n]["accuracy"]
                num_hackathon_manual_images[n] += trial_evaluation_df.iloc[n]["num_manual_labeled"] + trial_evaluation_df.iloc[n]["num_hackathon_labeled"]
        plt.plot(num_hackathon_manual_images/num_trials, accuracies/num_trials, label = experiment, linewidth = 2)
            
            
    xlab = "number manually labeled images"
    ylab = "test accuracy"
    title = "Prediction Accuracy vs Number of Manually Labeled Images"

    plt.xlabel(xlab,fontsize = 20, labelpad = 20)
    plt.ylabel(ylab, fontsize = 20, labelpad = 10)
    plt.title(title, fontsize = 26, pad = 20)

    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)



    plt.tick_params(length = 6, width = 3)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    #citizen_accuracy = get_citizen_accuracies()[0]

    #plt.hlines(y = citizen_accuracy, xmin = np.min(df["num_training_images"]), xmax = np.max(df["num_training_images"]), label = "citizen_accuracy", color = "red", linestyle = "--")  

    if(custom_legend_labels is None):
        plt.legend(fontsize = 10, bbox_to_anchor = (1,1))
    else:
        plt.legend(loc = "upper left",fontsize = 14, labels = custom_legend_labels)


    plt.tight_layout()

    if(savename is not None):
        plt.savefig(savename, dpi = 400)
    else:
        plt.savefig("./SSAL_outputs/" + "al_plot", dpi = 400)   
    
        
        
def make_al_plot(model_name, savename = None):
    
    

    path_name = "./SSAL_outputs/" + model_name + "/"
    file_name = path_name + model_name + "_evaluation_df.csv"

    df = pd.read_csv(file_name, index_col = 0)


    ax = df.plot(x = "num_training_images", y = ["accuracy","top2accuracy"],figsize = (14,8))

    xlab = "num_training_images"
    ylab = "metric value"
    title = "metrics vs number of training samples - " + model_name    

    plt.xlabel(xlab,fontsize = 20, labelpad = 20)
    plt.ylabel(ylab, fontsize = 20, labelpad = 10)
    plt.title(title, fontsize = 26)

    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)



    plt.tick_params(length = 6, width = 3)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    #citizen_accuracy = get_citizen_accuracies()[0]

    #plt.hlines(y = citizen_accuracy, xmin = np.min(df["num_training_images"]), xmax = np.max(df["num_training_images"]), label = "citizen_accuracy", color = "red", linestyle = "--")  

    plt.legend(fontsize = 10, bbox_to_anchor = (1,1))


    plt.tight_layout()

    if(savename is not None):
        plt.savefig(path_name + savename)
    else:
        plt.savefig(path_name + model_name + "_al_plot.png", dpi = 400)   
        
        
def make_al_dist_plot_experiment(experiment_list, savename = "None", fmt = "%d", dataset = "GLOBE", num_classes = 10, custom_legend_labels = None):

    all_model_al_dists = []
    experiment_dict = {}
    experiment_names = {}
    weird_c = []

    i = 0

    X = get_dataset_label_names(dataset = dataset, num_classes = num_classes)[1:]
    


    #for every experiment, average the distribution of labels in the final active learning set in every trial
    for experiment in experiment_list:
        print(experiment)
        experiment_params = pd.read_csv("./SSAL_outputs/" + experiment + "/" + experiment + "_params.csv", index_col = 0)
        num_trials = int(experiment_params[experiment].num_trials)
        
        experiment_dist = np.zeros(num_classes)
       
        for i in range(num_trials):
            trial_name = experiment + "_" + str(i) 
            trial_evaluation_df = pd.read_csv("./SSAL_outputs/" + experiment + "/" + trial_name + "/" + trial_name + "_evaluation_df.csv", index_col = 0)
            
            
            last_al_set_trial = ast.literal_eval(trial_evaluation_df["hackathon_set_composition_n"].iloc[-1])
            experiment_dist += last_al_set_trial


        experiment_dict[experiment] = experiment_dist/num_trials

        i += 1

    #print(all_model_accuracies)   

   


    df = pd.DataFrame(experiment_dict, index = X)

    title = "Active Learning Selection Compositions"
    xlab = "label"
    ylab = "number of instances"


    make_pandas_bar_plot(df, title = title, xlab = xlab, ylab = ylab, savename = savename, bar_label_fmt = fmt, custom_legend_labels = custom_legend_labels)

