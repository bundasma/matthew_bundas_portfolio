'''
Semi-supervised Active Learning Experiment Driver

Is the main class facilitating an SSAL experiment. If a parameter column is passed to an SSAL Driver
instance when it is created, it uses those parameters in the experiment, otherwise hard-coded values are used.

Just about everything outside of the actual CNN model creation, training and predictions happens in this class.

Each step of the SSAL experiment is layed out in a function or in several functions and executed one after the other.
A loop is used to run the SSAL experiment for a given number of trials. 

'''




from SSALModel import SSALModel
from SSAL_src.SSAL_evaluation import get_detailed_accuracy, get_detailed_f1, get_top_k_accuracies, fill_in_freq_gaps
from SSAL_src.SSAL_util import * 

from distutils.util import strtobool

import ast

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os 
import os.path
import time
import psycopg2 as pg
import psycopg2.extras
import pandas.io.sql as psql
import shutil

import warnings

import keras.backend as K

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import compute_class_weight

from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int32, addapt_numpy_int32)
register_adapter(np.int64, addapt_numpy_int64)





class SSALDriver:
    
    def __init__(self, param_col = None, experiment_name = None, trial_num = None):
        
        
        
        
        if(param_col is not None):
            
            self.name = str(experiment_name) + "_" + str(trial_num)
            
            self.onIteration = 0 #iteration driver starts on
            self.maxIterations = int(param_col.maxIterations) #number of iterations of workflow to complete
            
            self.numLLEpochs = int(param_col.numLLEpochs) #number of epochs to train just last layer
            self.fineTuneEpochs = int(param_col.fineTuneEpochs) #number of epochs to train entire model
            
            self.numDenseNeurons = float(param_col.numDenseNeurons) #number of neurons in dense layer
            self.labelSmoothing = float(param_col.labelSmoothing) #amount of label smoothing
            self.dropoutRate = float(param_col.dropoutRate) #dropout rate in noised models
            
            self.LLLearningRate = float(param_col.LLLearningRate) #learning rate when training just last layer
            self.fineTuneLearningRate = float(param_col.fineTuneLearningRate) #learning rate when training entire model
            
            self.batchSize = int(param_col.batchSize) #batch size for all training
            
            self.weightClasses = bool(strtobool(param_col.weightClasses)) #weight classes given original training dist
            
            self.lossFunction = string(param_col.lossFunction)
            
            self.augmentImages = bool(strtobool(param_col.augmentImages)) #augment images in training set
            self.useValidation = bool(strtobool(param_col.useValidation)) #use evaluation set as validation set
            self.earlyStoppingPatience = float(param_col.earlyStoppingPatience) #-1 for no early stopping
            
            
            self.writeFinalModel = bool(strtobool(param_col.writeFinalModel)) #when all done, write model
            self.modelType = param_col.modelType #inception or densenet
            
            self.database = str(param_col.datbase)#"intel"
            self.remoteDB = bool(strtobool(param_col.remoteDB)) #True/False use remote database or not
            self.dbName = param_col.dbName
            
            
            #used to tell program when to stop
            self._continue = True
            
            #size of evaluation set to put aside 
            
            self.optimizer= param_col.optimizer 
            
            #size of training set throughout all iterations, either a number of images or "full"
            if(type(param_col.TVPoolSize) == str):
                self.TVPoolSize = int(param_col.TVPoolSize.replace(',',''))#"full" #"full" for no limit, 500 in 500/2500 split
            else:
                self.TVPoolSize = int(param_col.TVPoolSize)            

            self.validationRatio =float(param_col.validationRatio)
            
            self.evaluationSize = int(param_col.evaluationSize)
            
            #size of prediction set, number of images
            self.predictionSize = int(param_col.predictionSize)
            
            #weight to give citizen predictions when boosting confidence
            self.citizenWeight = float(param_col.citizenWeight)
            
            
            
            #threshold for accepting predictions as "confident"
            self.confidentThreshold = float(param_col.confidentThreshold)
            
            #number of active learning selections to make at each iteration
            self.numALSamplesPerIter = int(param_col.numALSamplesPerIter)
            
            self.activeLearningLabeler = param_col.activeLearningLabeler #simulated or local_oracle, how we clear out active learning
            
            self.alSelectionCriteria = param_col.alSelectionCriteria #reg_entropy or mod_entropy or random, which version we use in active learning selections
            
            self.SSLType = param_col.SSLType#nonboosted, boosted, none
            self.pseudoLabelSelectionPerClass = int(param_col.pseudoLabelSelectionPerClass)
            
            self.classNoise = ast.literal_eval(param_col.classNoise)
            self.classImbalance = ast.literal_eval(param_col.classImbalance)
        
            self.trainOnCitizenLabels = bool(strtobool(param_col.trainOnCitizenLabels)) #augment images in training set
            self.preBalanceData = bool(strtobool(param_col.preBalanceData))
            print(self.classNoise)
            print(self.classImbalance)
        

            #create directories used to log/evaluate model
            self.saveDirectory = "./SSAL_outputs/" + experiment_name + "/" + self.name + "/"
        
        else:
        
        
            self.onIteration = 0 #iteration driver starts on
            self.maxIterations = 2 #number of iterations of workflow to complete
            
            self.numLLEpochs = 1 #number of epochs to train just last layer
            self.fineTuneEpochs = 0 #number of epochs to train entire model
            
            self.numDenseNeurons = 256*2 #number of neurons in dense layer
            self.labelSmoothing = 0 #amount of label smoothing
            self.dropoutRate = 0.5 #dropout rate in noised models
            
            self.LLLearningRate = 9e-5 #learning rate when training just last layer
            self.fineTuneLearningRate = 0.000001 #learning rate when training entire model
            
            self.batchSize = 32 #batch size for all training
            
            self.weightClasses = True #weight classes given original training dist
            self.augmentImages = False #augment images in training set
            self.useValidation = True#use evaluation set as validation set
            self.earlyStoppingPatience = 5# -1 for no early stopping
            self.writeFinalModel = False #when all done, write model
            
            self.modelType = "inception" #inception or densenet
            self.lossFunction = "MAE"
            
            self.database = "GLOBE"
            self.remoteDB = True #True/False use remote database or not
            self.dbName = "globe_images_6" #keep as 6 to avoid messing up live jobs
            
            #used to tell program when to stop
            self._continue = True
            
            self.optimizer = "adam" # "adam" or "sgd"
    
            self.TVPoolSize = 500
            self.validationRatio = 0.2
    
            
            #size of prediction set, number of images
            self.predictionSize = 1000
            self.evaluationSize = 1000
            
            #weight to give citizen predictions when boosting confidence
            self.citizenWeight = 0.75
            
            #threshold for accepting predictions as "confident"
            self.confidentThreshold = 0.80
            
            #number of active learning selections to make at each iteration
            self.numALSamplesPerIter = 50
            
            self.activeLearningLabeler = "local_oracle" #simulated or local_oracle, how we clear out active learning
            
            self.alSelectionCriteria = "mod_entropy" #reg_entropy or mod_entropy or random, which version we use in active learning selections
            
            self.SSLType = "boosted"#nonboosted, boosted, none
            self.pseudoLabelSelectionPerClass = 0 #-n for balanced with a built n, 0 for all, n for take n
            
            self.classNoise = [0, 0, 0, 0, 0, 0 ]
            self.classImbalance = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
            self.trainOnCitizenLabels = True
            self.preBalanceData = False
            
            #name of model
            self.name = str( np.random.randint(low = 0, high = 10000))
           
            
            #create directories used to log/evaluate model
            self.saveDirectory = "./SSAL_outputs/" + self.name + "/"
        
        #initialize other members that get assigned later
        self.model = None
        self.databaseConnection = None
        self.currentTrainingSet = None
        self.predictionResults = None
        self.evaluationSetDF = None
        self.citizenAccuracies = None #np.zeros(self.numClasses + 1)#[0,0,0,0,0,0,0,0,0,0,0]
        self.numManualLabeled = 0
        self.numModelLabeled = 0
        self.numHackathonLabeled = 0
        self.numWeb = 0
        self.confidentPredictions = pd.DataFrame()
        self.unconfidentPredictions = pd.DataFrame()
        self.pseudolabelSet = pd.DataFrame()
        self.numHackathonLabeledAfterWait = 0
        self.numHackathonLabeledBeforeWait = 0 
        self.trainingSetComposition_n = []
        self.pseudolabelSetComposition_n = []
        self.hacakthonSetComposition_n = []
    
        #columns we write at the end of every iteration
        self.evaluationColumns = ["iteration","num_training_images", "num_manual_labeled","num_model_labeled","num_hackathon_labeled", "num_web","num_confident","num_unconfident","num_pseudolabels_assigned","num_al_labels_recieved","training_set_composition_n","training_set_composition_per","pseudolabel_set_composition","hackathon_set_composition_n","accuracy","top2accuracy","f1","citizen_accuracies"]
        self.evaluationDF = pd.DataFrame(columns = self.evaluationColumns)
        
        #columns with type list need to be typed as object
        obj_cols = ["training_set_composition_n","training_set_composition_per"]
        
        for col in obj_cols:
            self.evaluationDF[col] = self.evaluationDF[col].astype(object)
        
        
        is_dir = os.path.isdir(self.saveDirectory)

        if(is_dir == False):
          
            os.mkdir(self.saveDirectory)
            os.mkdir(self.saveDirectory + "iteration_predictions/")
            os.mkdir(self.saveDirectory + "iteration_pseudolabels/")
            os.mkdir(self.saveDirectory + "iteration_active_learning_selections/")
            os.mkdir(self.saveDirectory + "iteration_histories/")
        self.save_metadata()

    #writes the metadata/parameters used in Driver
    def save_metadata(self):
        
        
        self.metaData_columns = ["name",         "maxIterations",    "numLLEpochs",     "fineTuneEpochs",    "numDenseNeurons",   "labelSmoothing",     "dropoutRate",    "LLLearningRate",    "fineTuneLearningRate",   "batchSize",    "weightClasses",    "augmentImages",   "useValidation",     "earlyStoppingPatience",    "writeFinalModel", "modelType"  ,     "remoteDB",    "dbName",    "optimizer",    "TVPoolSize",     "validationRatio",     "predictionSize",    "citizenWeight",    "confidentThreshold",     "numALSamplesPerIter",    "activeLearningLabeler",    "alSelectionCriteria",    "SSLType",     "classNoise",    "classImbalance",    "trainOnCitizenLabels", "preBalanceData"]
        self.metaData_data =    [[self.name, self.maxIterations, self.numLLEpochs, self.fineTuneEpochs, self.numDenseNeurons, self.labelSmoothing, self.dropoutRate, self.LLLearningRate, self.fineTuneLearningRate, self.batchSize, self.weightClasses, self.augmentImages, self.useValidation, self.earlyStoppingPatience, self.writeFinalModel, self.modelType, self.remoteDB, self.dbName, self.optimizer, self.TVPoolSize, self.validationRatio,  self.predictionSize, self.citizenWeight, self.confidentThreshold, self.numALSamplesPerIter, self.activeLearningLabeler, self.alSelectionCriteria,   self.SSLType,  self.classNoise,  self.classImbalance,self.trainOnCitizenLabels, self.preBalanceData]]        
        metadata_df = pd.DataFrame(columns = self.metaData_columns, data = self.metaData_data)
        metadata_df.to_csv(self.saveDirectory + self.name + "_metadata.csv")
        
        
        
    #creates new model    
    def initialize_noised_model(self):
        
        self.model = SSALModel(SSALDriver = self)
        self.model.initializeNoisedTFModel()
        
    #creates new model    
    def initialize_denoised_model(self):
        
        self.model = SSALModel(SSALDriver = self)
        self.model.initializeDeNoisedTFModel()
        
    #evlautes whether the cycle should continue    
    def evaluate_continue(self):
        
        if(self.onIteration >= self.maxIterations):
            
            return False
        
        else:
            
            return True
        
    #connects to a postgres database    
    def connect_to_db(self):
        
        if(self.remoteDB == True):
            host = "exp2-plab.cs.nmsu.edu"
            dbname = self.dbName
            user = "preeti"
            password = "NasaEpscor01"
            
        else:
            host = "localhost"
            dbname = self.dbName
            user = "postgres"
            password = "J@nuary217"
        
        self.databaseConnection = pg.connect(
            
                host = host,
                dbname = dbname,
                user = user,
                password = password
              
            )
    #sets the number of classes based on database contents    
    def query_num_classes(self):
        
        if(self.trainOnCitizenLabels == True):
            sql_num_classes = """SELECT COUNT(DISTINCT citizen_label) FROM observation;"""
            
        else:
            sql_num_classes = """SELECT COUNT(DISTINCT dataset_label) FROM observation;"""

        num_classes = psql.read_sql(sql_num_classes, self.databaseConnection)["count"][0]
        
        self.numClasses = num_classes
    
    #gets all labeled data
    def get_full_labeled_data(self):
        
        
        
        sql_training_data = """SELECT * FROM observation WHERE official_label IS NOT NULL and partition IS DISTINCT FROM 'test' AND available=1;"""
            
        
        training_data = psql.read_sql(sql_training_data, self.databaseConnection)
        
        return training_data

        
    #sets evaluation set, a portion of the labeled data not to be used for training
    def get_evaluation_set(self):
        
        sql_evaluation_set = """SELECT * from observation where partition='test' AND available=1;"""
        
        evaluation_set = psql.read_sql(sql_evaluation_set, self.databaseConnection)
        
        
        evaluation_set = evaluation_set.sample(n = self.evaluationSize, replace = False)
        
        
        self.evaluationSetDF = evaluation_set
        
        self.evaluationSetDF.to_csv(self.saveDirectory + self.name + "_evaluation_set.csv")
    
  
    #injects noise into the labels based on self.classNoise    
    def inject_noise(self):
        
        if(self.database == "GLOBE" or ("globe" in self.dbName)):
            
            print("do not want to inject noise in globe")
            return
        
        
        
        sql_update_citizen_label = """UPDATE observation SET citizen_label = %s WHERE image_name = %s;"""
        
        
        cur = self.databaseConnection.cursor()
        
        #for each class, get all images, get random sample of x% according to classNoise, randomly pick a class for them
        for label_class in range(self.numClasses):
            
            print("injecting noise class,", label_class)
            
            #gets all instances of class
            sql_get_class = "SELECT * FROM observation where dataset_label=" + str(label_class)     
            class_df = psql.read_sql(sql_get_class, self.databaseConnection)
            
            #gets the instances needed to be noised
            noised_sample = class_df.sample(frac = self.classNoise[label_class], replace = False)
            
            #generates the noised labels 
            random_labels = np.random.randint(self.numClasses, size = noised_sample.shape[0])
            
            #updates database with noised labels
            rows = zip(random_labels,noised_sample.image_name)
            cur = self.databaseConnection.cursor()
            psycopg2.extras.execute_batch(cur,sql_update_citizen_label, rows)
            
        
        self.databaseConnection.commit()
    
    #simulates imbalance in dataset based on self.classImbalance
    def inject_imbalance(self):
        
        #don't want to inject noise into GLOBE dataset, it's already noised enough
        if(self.database == "GLOBE" or ("globe" in self.dbName)):
            
            print("do not want to inject imbalance in globe")
            return
        
        sql_update_availability = """UPDATE observation SET available = 0 WHERE image_name = %s;"""
        cur = self.databaseConnection.cursor()
        
        #for each class, get images from class, randomly select 1-value from classImbalance non-test images, switch their availability to 0
        #for each class, makes a subset "unavailable" in the database
        for label_class in range(self.numClasses):
            
            #1 means no imbalance
            if(self.classImbalance[label_class] == 1):
                continue
            
            print("injecting imbalance class,", label_class)
            
            #gets instances of class
            sql_get_class = "SELECT * FROM observation where dataset_label=" + str(label_class) + "AND partition IS DISTINCT FROM 'test' and available=1;"
            class_df = psql.read_sql(sql_get_class, self.databaseConnection)
            
            #gets subset of class to become unavailablee
            unavailable_sample = class_df.sample(frac = 1 - self.classImbalance[label_class], replace = False)
            
            #updates remote database
            rows = zip(unavailable_sample.image_name,)
            cur = self.databaseConnection.cursor()
            psycopg2.extras.execute_batch(cur,sql_update_availability, rows)
         
        self.databaseConnection.commit()
        
        
                
                
               
            
    #gets all the data not in the test partition and with a dataset label
    #splits it into train/valid
    #if training on citizen labels, assigns official label to citizen label
    #else assigns official label as the dataset label    
    def set_train_validate_pool(self):
        
        #get subset of images that will serve as train/validate pool
        if(self.trainOnCitizenLabels == True):
            sql_non_test_images ="""SELECT * FROM observation where partition IS DISTINCT FROM 'test' AND available=1 AND citizen_label IS NOT NULL;"""
        else:
            sql_non_test_images ="""SELECT * FROM observation where partition IS DISTINCT FROM 'test' AND available=1 AND dataset_label IS NOT NULL;"""

        df_non_test_images = psql.read_sql(sql_non_test_images, self.databaseConnection)
        df_train_valid_pool = df_non_test_images.sample(n=self.TVPoolSize, replace = False)
        
           
        #split into train/validate datasets
        df_train, df_valid = train_test_split(df_train_valid_pool, test_size = self.validationRatio)
        
        print("starting update")
        time_before_update_train = time.time()
        
        #for training set, update image's official_label, official_label_source to manual, partition to training
        if(self.trainOnCitizenLabels == True):
            sql_update_train_set = """UPDATE observation SET official_label=citizen_label, official_label_source = 'manual', partition='train' WHERE image_name = %s"""
        else:
            sql_update_train_set = """UPDATE observation SET official_label=dataset_label, official_label_source = 'manual', partition='train' WHERE image_name = %s"""
        
        #update database
        rows = zip(df_train.image_name,)
        cur = self.databaseConnection.cursor()
        psycopg2.extras.execute_batch(cur,sql_update_train_set, rows)
        self.databaseConnection.commit()
        
        time_after_update_train = time.time()
        
        print("time for updating train", time_after_update_train - time_before_update_train)
        
        time_before_update_valid = time.time()
        #for validation set, update image's official_label, official_label_source to manual, partition to validation
        if(self.trainOnCitizenLabels == True):
            sql_update_valid_set = """UPDATE observation SET official_label=citizen_label, official_label_source = 'manual', partition='validation' WHERE image_name = %s"""
        else:
            sql_update_valid_set = """UPDATE observation SET official_label=dataset_label, official_label_source = 'manual', partition='validation' WHERE image_name = %s"""
        
        #update database
        rows = zip(df_valid.image_name,)
        cur = self.databaseConnection.cursor()
        psycopg2.extras.execute_batch(cur,sql_update_valid_set, rows)
        self.databaseConnection.commit()
        
        time_after_update_valid = time.time()
        
        print("time for updating valid", time_after_update_valid - time_before_update_valid)
        
        self.validationSetDF = self.get_full_validation_set()
        
        #save original validation set
        self.validationSetDF.to_csv(self.saveDirectory + self.name + "_og_valid_set.csv")
        df_train.to_csv(self.saveDirectory + self.name + "_og_train_set.csv")
        
       
        
        
    #retrieves training set from database
    def get_full_training_set(self):
        
        sql_train = """SELECT * from observation WHERE partition='train' AND available=1;"""
        
        return psql.read_sql(sql_train, self.databaseConnection)
    
    #retrieves validation set from database
    def get_full_validation_set(self):
        
        sql_valid = """SELECT * from observation WHERE partition='validation' AND available=1;"""
        
        return psql.read_sql(sql_valid, self.databaseConnection)
        
    #sets the training set, sets meta-info about training set
    def set_current_training_set(self):
        
        #get full training set
        full_training_data = self.get_full_training_set()
        self.currentTrainingSet = full_training_data
        
        #get columns regarding training set
        official_label_sources = self.currentTrainingSet["official_label_source"].value_counts().sort_index()
        training_set_composition = self.currentTrainingSet["official_label"].value_counts().sort_index()
        hackathon_set_composition = self.currentTrainingSet[self.currentTrainingSet["official_label_source"] == "hackathon"]["official_label"].value_counts().sort_index()
      
 
        
        #number of manual labeled
        try:
            self.numManualLabeled = official_label_sources["manual"]
        except:
            self.numManualLabeled = 0
        
        #number of model labeled
        try:
            self.numModelLabeled = official_label_sources["model"]
        except:
            self.numModelLabeled = 0
        
        #number of hackathon labeled
        try:
            self.numHackathonLabeled = official_label_sources["hackathon"]
        except:
            self.numHackathonLabeled = 0
        
      
        
        #training set composition
        self.trainingSetComposition_n = training_set_composition.values
        self.hackathonSetComposition_n = fill_in_freq_gaps(hackathon_set_composition, num_classes = self.numClasses)
        
        #weight classes during training
        if(self.weightClasses == True):
            
            unique_labels = np.arange(self.numClasses)
            labels = full_training_data["official_label"]

            weights = compute_class_weight("balanced", unique_labels, labels)
            self.classWeightsDict = {i : weights[i] for i in range(len(weights))}
            
        else:
            self.classWeightsDict = None
        
        #saves original training set if we are on the first iteration
        if(self.onIteration == 0):
            self.currentTrainingSet.to_csv(self.saveDirectory + self.name + "_og_train_set.csv")
        
        
    
    #sets the current prediction set, a sample of the data w/o official label
    def set_current_prediction_set(self):
        
        #gets all data that could serve as prediction data (not in train or test partition)
        predictions_sql = """SELECT * FROM observation WHERE partition IS NULL AND available=1 AND image_name NOT LIKE '%web%'"""
        all_possible_predictions = psql.read_sql(predictions_sql, self.databaseConnection)    
        
        num_unlabeled = all_possible_predictions.shape[0]
        
        #gets sample of possible prediction set to serve as actual prediction set
        if(num_unlabeled >= self.predictionSize):
            
            current_prediction_set = all_possible_predictions.sample(n=self.predictionSize, replace = False)
        
        else:
            
            current_prediction_set = all_possible_predictions.sample(n=num_unlabeled, replace = False)
        
        self.currentPredictionSet = current_prediction_set
    
    #train the current model given the current training dataset
    def train_model(self, num_epochs = 1, batch_size = 32):
        
        self.model.trainTFModel(num_epochs = num_epochs, batch_size = batch_size)
        
    #make predictions given the current model and prediction set    
    def make_predictions(self, batch_size = 32):
        
        self.model.predictTFModel_prediction_set(batch_size = batch_size)
        
        #calculate entropy for later, before we start messing with probabilities
        self.predictionResults["reg_entropy"] = self.predictionResults.apply(reg_ent_df_func, axis = 1, args = (self.numClasses,))
        self.predictionResults["mod_entropy"] = self.predictionResults.apply(mod_ent_df_func, axis = 1, args = (self.trainingSetComposition_n/np.sum(self.trainingSetComposition_n), self.numClasses,))        
        
    #saves predictions, info about predictions before and after confidence boosting     
    def save_iteration_predictions(self):
        
        
        predictions_save_name = self.saveDirectory + "iteration_predictions/" + self.name + "_predictions_" + str(self.onIteration) + ".csv"
        #pre-boost prediction stats
        self.preBoostedPredictions["pre_boosted_confidence"] = self.preBoostedPredictions.apply(self.max_confidence, axis = 1)
        self.preBoostedPredictions["pre_boosted_confident"] = self.preBoostedPredictions.apply(self.is_confident, axis = 1)
        self.preBoostedPredictions["pre_boosted_prediction"] = self.preBoostedPredictions.apply(self.predicted_label, axis = 1)

        #post-boost prediction stats
        self.postBoostedPredictions["post_boosted_confidence"] = self.postBoostedPredictions.apply(self.max_confidence, axis = 1)
        self.postBoostedPredictions["post_boosted_confident"] = self.postBoostedPredictions.apply(self.is_confident, axis = 1)
        self.postBoostedPredictions["post_boosted_prediction"] = self.postBoostedPredictions.apply(self.predicted_label, axis = 1)

        predictions_df = pd.concat([self.preBoostedPredictions[["image_name","citizen_label","dataset_label","pre_boosted_confidence","pre_boosted_confident","pre_boosted_prediction"]], self.postBoostedPredictions[["post_boosted_confidence","post_boosted_confident","post_boosted_prediction", "0_prob", "1_prob", "2_prob", "3_prob", "4_prob", "5_prob"]]], axis = 1)
        
        #none of them have an official label yet
        predictions_df["official_label"] = np.nan
       
        #identifies if a prediction will be used as a pseudolabel or not
        if(self.SSLType != "none" and self.pseudolabelSet.shape[0] > 0):
            predictions_df["used_as_pseudolabel"] = predictions_df.image_name.isin(self.pseudolabelSet.image_name).astype(int)
        else:
            predictions_df["used_as_pseudolabel"] = 0
        
        predictions_df.to_csv(predictions_save_name)
    
   
    #grabs citizen labels, adds them to prediction set    
    def add_citizen_labels_to_predictions(self):
        
        print("adding citizen labels to predictions")
        
        citizen_labels = []
        
        for index, row in self.predictionResults.iterrows():
            
            image_name = row["image_name"]
            og_row = self.currentPredictionSet[self.currentPredictionSet["image_name"] == image_name
            citizen_label = og_row["citizen_label"].iloc[0]
            citizen_labels.append(citizen_label)
           
        
        self.predictionResults["citizen_label"] = citizen_labels
    
    #grabs citizen labels, adds them to prediction set    
    def add_dataset_labels_to_predictions(self):
        
        print("adding dataset labels to predictions")
        
        dataset_labels = []
        
        for index, row in self.predictionResults.iterrows():
            
            image_name = row["image_name"]
            og_row = self.currentPredictionSet[self.currentPredictionSet["image_name"] == image_name]
            dataset_label = og_row["dataset_label"].iloc[0]
            dataset_labels.append(dataset_label)
           
        
        self.predictionResults["dataset_label"] = dataset_labels    
    
    #grabs citizen labels, adds them to evaluation set
    def add_citizen_labels_to_evaluations(self):
        
        print("adding citizen labels ")
        
        citizen_labels = []
        
        for index, row in self.evaluationPredictionResults.iterrows():
            
            image_name = row["image_name"]
            og_row = self.evaluationSetDF[self.evaluationSetDF["image_name"] == image_name]
            citizen_label = og_row["citizen_label"].iloc[0]
            citizen_labels.append(citizen_label)
           
        self.evaluationPredictionResults["citizen_label"] = citizen_labels
        
    #boosts the predictions/probabilities given citizen labels     
    def boost_row(self,row):

        #get the citizen label
        citizen_label = row["citizen_label"]

        if(pd.isna(citizen_label)):
            
            return row
                                               
        citizen_label = int(citizen_label)
                                               
        #get the citizen accuracy
        citizen_accuracy = self.citizenAccuracies[citizen_label]

        #get the probability the model spat out for the citizen label
        model_probability = row.iloc[citizen_label]
        
        #calculate new probability
        new_probability = citizen_accuracy*self.citizenWeight + model_probability - (citizen_accuracy * self.citizenWeight * model_probability)
        
        #set the new probability
        row.iloc[citizen_label] = new_probability
        
        return row
        
    #boost predictions given citizen labels    
    def boost_predictions(self):
        
        self.preBoostedPredictions = self.predictionResults.copy()
        if(self.SSLType == "boosted"):
            self.predictionResults = self.predictionResults.apply(self.boost_row, axis = 1)
        
        self.postBoostedPredictions = self.predictionResults.copy()
        
    #determines whether a row is confident prediction or not    
    def is_confident(self,row):
        
        probabilities = row.iloc[0:self.numClasses]
        
        if(np.max(probabilities) >= self.confidentThreshold):
            
            return 1
        
        else:
            
            return 0
        
    def max_confidence(self,row):
        
        probabilities = row.iloc[0:self.numClasses]
        
        return np.max(probabilities)
    
    def predicted_label(self, row):
        
        probabilities = row.iloc[0:self.numClasses]
        
        return np.argmax(probabilities)

        
        
    #separates confident (potentially to be pseudolabeled), unconfident (to be active learned)    
    def split_confident_unconfident(self):
        
        self.predictionResults["confident"] = self.predictionResults.apply(self.is_confident, axis = 1)
        
        self.confidentPredictions = self.predictionResults[self.predictionResults["confident"] == 1]
        
        self.unconfidentPredictions = self.predictionResults[self.predictionResults["confident"] == 0]
        
        #print("total predictions", self.predictionResults.shape[0])
        #print("confident predictions", self.confidentPredictions.shape[0])
        #print("unconfident predictions",self.unconfidentPredictions.shape[0])
        
    #separates confident (to be labeled), unconfident (to be active learned)    
    def select_pseudolabels(self):
        
        
        #if there are no confident predictions, assign no pseudolabels
        if(self.confidentPredictions.shape[0] == 0):
            
            print("no confident predictions")
            
            self.pseudolabelSet = pd.DataFrame()
            self.pseudolabelSetComposition_n = np.zeros(self.numClasses)#[0,0,0,0,0,0,0,0,0,0,0]
            
       
            self.pseudolabelSet.to_csv(self.saveDirectory + "iteration_pseudolabels/" + self.name + "_pseudolabels_" + str(self.onIteration) + ".csv")
        
            return
        #if we don't want to have any semi-supervised learning, assign no pseudolabels
        if(self.SSLType == "none"):
            
            self.pseudolabelSet = pd.DataFrame()
            self.pseudolabelSetComposition_n = np.zeros(self.numClasses)#[0,0,0,0,0,0,0,0,0,0,0]
            return
        
        #get pre-pseudolabeling class distribution                                       
        current_class_distribution = self.trainingSetComposition_n
        
        #get confident distribution                                       
        confident_dist = self.confidentPredictions["predicted_label"].value_counts().sort_index()
        
        print("confident_dist fill")
        print(confident_dist)
    
        #dataframe for final pseudolabels                                       
        final_pseudolabels = pd.DataFrame()
        
        #takes number of pseudolabels according to self.pseudoLabelSelectionPerClass                                       
        #-n for balanced with a built n, 0 for all, n for take n
        
        #take distribution that won't make balancing worse                                       
        if(self.pseudoLabelSelectionPerClass < 0):
            confident_dist_filled = fill_in_freq_gaps(confident_dist, fill = np.abs(self.pseudoLabelSelectionPerClass), num_classes = self.numClasses)
            take = calculate_correct_end_population(current_class_distribution, confident_dist_filled, num_classes = self.numClasses)
                                               
        #take all pseudolabels we can                                       
        elif(self.pseudoLabelSelectionPerClass == 0):
            take = np.full(self.numClasses, self.predictionSize)
                                               
        #take flat number of available pseudolabels     
        else:
            take = np.full(self.numClasses, self.pseudoLabelSelectionPerClass)
            
        #uses selected distibution to determine final pseudolabels
        for label_class in range(self.numClasses):
            
            #get confident predictions for class
            class_predictions = self.confidentPredictions[self.confidentPredictions["predicted_label"] == label_class]
            
            #sort based on entropy
            entropy_sorted = class_predictions.sort_values(by = ["reg_entropy"])
            
            #take allowed # of class, sorted by entropy
            take_class = take[label_class]
            
            keep_sample = entropy_sorted.iloc[:take_class,:]

            final_pseudolabels = final_pseudolabels.append(keep_sample, ignore_index = True)
           
        
        self.pseudolabelSet = final_pseudolabels
        self.pseudolabelSetComposition_n = fill_in_freq_gaps(final_pseudolabels["predicted_label"].value_counts().sort_index(), num_classes = self.numClasses)
        self.pseudolabelSet["official_label"] = np.nan
        
        print("pseudolabelset composition")
        print(self.pseudolabelSetComposition_n)
        
        #save pseudolabel selection                                       
        self.pseudolabelSet.to_csv(self.saveDirectory + "iteration_pseudolabels/" + self.name + "_pseudolabels_" + str(self.onIteration) + ".csv")
        
        
        
        
    #determines the subset of unconfident predictions to be active learned    
    def select_al_subset(self):

        #calculate active learning selection criteria
        if(self.alSelectionCriteria == "mod_entropy"):
            self.unconfidentPredictions = self.unconfidentPredictions.sort_values(by = ["mod_entropy"], ascending = False)
        elif(self.alSelectionCriteria == "reg_entropy"):
            self.unconfidentPredictions = self.unconfidentPredictions.sort_values(by = ["reg_entropy"], ascending = False)
        else:
            #if random, keep all unconfident predictions, shuffle them for selection later
            self.unconfidentPredictions = self.unconfidentPredictions.sample(frac = 1, replace = False)
            
        #get active learning subset
        self.alSubset = self.unconfidentPredictions.iloc[0:self.numALSamplesPerIter,:]
        self.alSubset.reset_index(inplace = True, drop = True)
        self.alSubset.to_csv(self.saveDirectory + "iteration_active_learning_selections/" + self.name + "_alset_" + str(self.onIteration) + ".csv")

    
    
    #update database with confident predictions    
    def beam_up_pseudolabel_predictions(self):
        
        #do nothing with pseudolabels if told so                                       
        if(self.SSLType == "none" or self.pseudolabelSet.shape[0] == 0):
            
            return
        
        #updates database with pseudolabels
        cur = self.databaseConnection.cursor()       
        sql_update_pseduolabel = """UPDATE observation SET official_label = %s, official_label_source = 'model', partition = 'train' WHERE image_name =%s;"""
        rows = zip(self.pseudolabelSet.predicted_label, self.pseudolabelSet.image_name)
        cur.executemany(sql_update_pseduolabel, rows)
        self.databaseConnection.commit()
        
        cur.close()
    
   
    #updates database with active learning selections    
    def label_al_selection(self):
        
        if(self.database == "intel"):
            #for observation in alSubset, set official_label to dataset_label, official_label_source to hackathon, partition to train
            sql_update_al = """UPDATE observation set official_label=dataset_label, official_label_source='hackathon', partition='train' WHERE image_name = %s"""
            rows = zip(self.alSubset.image_name,)
            
            cur = self.databaseConnection.cursor()
            cur.executemany(sql_update_al, rows)
            
            self.databaseConnection.commit()
            
            cur.close()
        elif(self.database == "globe" or self.database == "GLOBE"):
            
            #beam up active learning selection
            print("beaming up active learning selections")
            
            al_ranks = np.arange(self.alSubset.shape[0])         
            sql_update_al = """UPDATE observation set al_rank = %s WHERE image_name = %s"""
            rows = zip(al_ranks, self.alSubset.image_name)
            cur = self.databaseConnection.cursor()
            cur.executemany(sql_update_al, rows)
            self.databaseConnection.commit()
            cur.close()
            
            #wait for active learning queue to be empty
            sql_al_queue = """SELECT * FROM observation WHERE al_rank IS NOT NULL"""
            
            al_queue_df = psql.read_sql(sql_al_queue, self.databaseConnection)
            
            while(al_queue_df.shape[0] > 0):
                
                print("al queue is not empty, waiting 30s")
                time.sleep(30)
                al_queue_df = psql.read_sql(sql_al_queue, self.databaseConnection)
            
            print("al queue is empty, moving on to next step")
                
                
        
        
        

 
    #evaluates citizen accuracy given observations with actual label assigned
    def evaluate_citizen_accuracy(self):
        
        labeled_data = self.get_full_labeled_data()
        labeled_data = labeled_data[labeled_data["citizen_label"].notna()]
        
        total_accuracy = accuracy_score(labeled_data["official_label"],labeled_data["citizen_label"])
        
        conf_matrix = confusion_matrix(labeled_data["official_label"], labeled_data["citizen_label"])
        
        byclass_accuracies = (conf_matrix.diagonal()/conf_matrix.sum(axis=1)).tolist()
        
        byclass_accuracies.append(total_accuracy)
        
        self.citizenAccuracies = byclass_accuracies
    
    #writes information about the iteration to csv                                           
    def write_iteration(self):
        
        self.iterationRow = pd.Series(index = self.evaluationColumns, dtype = object)
        #print(self.iterationRow)
        self.iterationRow["iteration"] = self.onIteration
        
        
        self.iterationRow["num_training_images"] = self.currentTrainingSet.shape[0]
        self.iterationRow["num_manual_labeled"] = self.numManualLabeled
        self.iterationRow["num_model_labeled"] = self.numModelLabeled
        self.iterationRow["num_hackathon_labeled"] = self.numHackathonLabeled
        self.iterationRow["num_web"] = self.numWeb
        
        self.iterationRow["num_confident"] = self.confidentPredictions.shape[0]
        self.iterationRow["num_unconfident"] = self.unconfidentPredictions.shape[0]
        self.iterationRow["num_pseudolabels_assigned"] = self.pseudolabelSet.shape[0]
        self.iterationRow["num_al_labels_recieved"] = self.numHackathonLabeledAfterWait - self.numHackathonLabeledBeforeWait
        
        self.iterationRow["training_set_composition_n"] = self.trainingSetComposition_n
        self.iterationRow["training_set_composition_per"] = self.trainingSetComposition_n/self.currentTrainingSet.shape[0] * 100
        self.iterationRow["pseudolabel_set_composition_n"] = self.pseudolabelSetComposition_n
        self.iterationRow["hackathon_set_composition_n"] = self.hackathonSetComposition_n
        
        self.iterationRow["accuracy"] = self.accuracy
        self.iterationRow["top2accuracy"] = self.top2Accuracy
        self.iterationRow["f1"] = self.f1
        
        self.iterationRow["citizen_accuracies"] = self.citizenAccuracies
        
        self.evaluationDF = self.evaluationDF.append(self.iterationRow, ignore_index = True)
        
        self.evaluationDF.to_csv(self.saveDirectory + self.name + "_evaluation_df.csv")
    
    
    
    #evaluates the model given the current model, evaluation set
    def evaluate_model(self):
        
        
        pre_eval = time.time()
        
        #make predictions                                       
        self.model.predictTFModel_evaluation_set()

        y_pred = self.evaluationPredictionResults["predicted_label"]
        y_true = self.evaluationPredictionResults["official_label"]
        
        #get probabilities                                       
        prob_list = get_prob_list(self.numClasses)
        cols = prob_list
        cols.append("predicted_label")
        cols.append("official_label")
        
        y_per_preds = self.evaluationPredictionResults[cols]
         
        #get metrics                                       
        self.accuracy = get_detailed_accuracy(y_true, y_pred)[0]
        self.top2Accuracy = get_top_k_accuracies(y_per_preds, 2, num_classes = self.numClasses)[0]
        self.f1 = get_detailed_f1(y_true, y_pred)[0]
        
        post_eval = time.time()
        
        
        print("time for evaluation", post_eval - pre_eval)
    
    #performs a few operations that need to take place at the end of iterations
    def wrap_up(self):

        #add citizen labels to final evaluation predictions        
        self.add_citizen_labels_to_evaluations()

        #save final evaluation predictions
        self.evaluationPredictionResults.to_csv(self.saveDirectory + self.name + "_final_preds.csv")

        #save final model
        if(self.writeFinalModel == True):
            
            self.model.TFModel.save(self.saveDirectory + self.name + "_model")

        self.databaseConnection.close()
        
    def do_post_iteration_adjustments(self):
        
        return
    
    #resets the database to be ready for starting experiment                                           
    def reset_database(self):
        
       
        cur = self.databaseConnection.cursor()
              
        #set all availability to 1 - remove class imbalance
        sql_set_avail_1 = """UPDATE observation SET available=1;"""
        cur.execute(sql_set_avail_1,())
        
        #set all citizen_labels to dataset_label - remove label noise                                       
        if(self.database != "globe" and self.database != "GLOBE" and "globe" not in self.dbName and self.database == "intel"):
            
            sql_reset_cit_label = """UPDATE observation SET citizen_label = dataset_label;"""
            cur.execute(sql_reset_cit_label,())

        
        #reset all non-test data, remove official labels, reset partition
        sql_reset_non_test = """UPDATE observation SET official_label=NULL, partition=NULL,official_label_source=NULL WHERE partition IS DISTINCT FROM 'test'"""
        cur.execute(sql_reset_non_test, ())
        
        self.databaseConnection.commit()
            
        #pre-balance non-test data to all start with 2190 images
        self.pre_balance_train_valid()
        
    #gets pre-balanced, raw/original label counts  
    def get_pre_balanced_label_counts(self):
        
        #the number of each label in non-test partition
        sql_label_counts = """SELECT dataset_label, COUNT(dataset_label) FROM observation WHERE PARTITION is DISTINCT FROM 'test' GROUP BY dataset_label;"""
    
        df_label_counts = psql.read_sql(sql_label_counts, self.databaseConnection)["count"].values
        
        return df_label_counts
        
       
    #forces class imbalance by underampling dominant classes
    def pre_balance_train_valid(self):
        
        #only want to pre-balance intel experiments                                       
        if(self.database == "GLOBE" or ("globe" in self.dbName)):
            
            print("do not want to pre-balance globe")
            return
        
        
        curs = self.databaseConnection.cursor()
        
        #gets pre-balanced counts                                       
        label_counts = self.get_pre_balanced_label_counts()
        min_count = np.min(label_counts)
        
        sql_update_availability = """UPDATE observation SET available = 0 WHERE image_name = %s;"""
        
        #for each class, select 2190 images to keep
        for label_class in range(self.numClasses):
            
            print("pre-balancing class,", label_class)

            num_make_unavail = label_counts[label_class] - min_count
            
            print(num_make_unavail)
          
            sql_get_class = "SELECT * FROM observation where partition IS DISTINCT FROM 'test' AND available = 1 AND dataset_label=" + str(label_class)
            class_df = psql.read_sql(sql_get_class, self.databaseConnection)
            unavail_sample = class_df.sample(n = num_make_unavail, replace = False)
                                               
            rows = zip(unavail_sample.image_name,)
            psycopg2.extras.execute_batch(curs,sql_update_availability, rows)
        
        self.databaseConnection.commit()    
    
    #coordinates a full trial of an experiment, given its parameters                                           
    def run_full_trial(self):
        
        print("running full trial")
        
        print("driver name", self.name)


        print("connecting to db")
        self.connect_to_db()
        
        print("querying number of classes")
        self.query_num_classes()
        
        print("resetting database")
        self.reset_database()
        
        print("injecting noise")
        self.inject_noise()
        
        print("injecting imbalance")
        self.inject_imbalance()
        
        print("setting train/validate pool in database")
        self.set_train_validate_pool()
        
        print("getting evaluation set")
        self.get_evaluation_set()
    
        print("entering loop")
        while(self.evaluate_continue() == True):
            
            print("starting iteration",self.onIteration)
            
            
            print("getting training set")
            self.set_current_training_set()
            
            print("getting prediction set")
            self.set_current_prediction_set()
            
            print("creating new model")
            self.initialize_noised_model()

            print("training model")
            self.train_model()
            
            print("making predictions")
            self.make_predictions()
            
            print("evaluating citizen accuracy")
            self.evaluate_citizen_accuracy()
            
            print("boosting predictions")
            self.boost_predictions()
            
        
            print("splitting confident/unconfident")
            self.split_confident_unconfident()
            
            print("selecting pseudolabel subset")
            self.select_pseudolabels()
            
            print("saving iteration predictions/pseudolabels")
            self.save_iteration_predictions()
            
            print("selecting AL subset")
            self.select_al_subset()
            
            print("beaming up confident predictions")
            self.beam_up_pseudolabel_predictions()
            
            print("label al selection")
            self.label_al_selection()
            
            print("evaluating model")
            self.evaluate_model()
            
         
            print("writing iteration")
            self.write_iteration()   
            
            print("doing post iteration adjustments")
            self.do_post_iteration_adjustments()
            
            print("done with iteration", self.onIteration,"\n\n\n")
            
            self.onIteration += 1
            
        self.wrap_up()
        
 
            
            
#test_driver = SSALDriver()
#test_driver.connect_to_db()
#test_driver.query_num_classes()

#test_driver.reset_database()
#test_driver.inject_noise()
#test_driver.inject_imbalance()
#test_driver.set_train_validate_pool()
#test_driver.run_full_trial()
    


