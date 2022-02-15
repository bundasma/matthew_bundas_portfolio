This folder contains much of the code I wrote for my research project surrounding the NASA GLOBE Observer Program. Details about the project are in the folder one level higher up. 

SSALDriver.py - Main driver of SSAL experiments. Attributes are experiment parameters/metrics/datasets. Functions facilitate semi-supervised active learning, communicating with the external postgres database. The bulk of the actual code is found in this class.

SSALModel.py - Class containing the actual tensorflow models, handles training, predicting, testing etc. 

SSAL_workbench.py - Helps automate running of many experiments, reading from a parameter csv file. 

SSAL_src/ - Contains the source files for some external functions. For the most part, they are evaluation functions used to evaluate the experiments in post.
  SSAL_classification_vis.py - Some functions to facilitate visualizing model classifications. Displays images, their actual class, and predicted class.
  SSAL_evaluation.py - Much of the functions actually calculating evlaution metrics, as well as some plotting.
  SSAL_models.py - Related to the models/datassets themselves. Constructing, compliling models and datasets.
  SSAL_plotting.py - Some meta plotting functions to help create evaluation plots in conjunction with SSAL_evaluation.py functions.
  SSAL_util.py - General purpose functions, entropy calculations, name list retrieving etc. 
  
  

SSAL_evaluation.ipynb - Evaluation of specified SSAL experiments. Printing pseudolabel performance, training set composition, Active Learning selections etc.

SSAL_experiment_csv_constructor.ipynb - Helps create experiment parameter csvs to feed to SSAL_workbench.py

SSAL_manual_labeling_1.ipynb - Facilitates reading from external postgres database, pulling down active learning selections and manually providing labels for them.

SSAL_param_search_eval.ipynb - Evaluation of paramater searching experiments, finding learning rates, epochs, early stopping parameters etc. Primarily plots learning curves.

SSAL_pseudolabel_labeling.ipynb - Providing labels to images SSAL experiments used as pseudolabels, so we can see if they were correct.

globe_presentation_plots.ipynb - Creation of SSAL evaluation plots for all relevant GLOBE SSAL experiments.

globe_presentation_plots.ipynb - Creation of SSAL evaluation plots for all relevant intel SSAL experiments.


