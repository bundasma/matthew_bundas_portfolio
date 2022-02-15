'''
Semi-supervised Active Learning Model

Defines the SSALModel class used in the SSALDriver to hold model-related information and carry out training, predictions etc

Requires an SSALDriver attribute to tell the model what it's supposed to do.

'''




from SSAL_src.SSAL_models import *
from SSAL_src.SSAL_util import *
import math
import time
import shutil
import keras.backend as K

from keras.callbacks import EarlyStopping
class SSALModel:
    
    #initialize the SSALModel
    def __init__(self, SSALDriver = None):
        
        self.TFModel = None
        self.modelHistory = None
        
        
        self.SSALDriver = SSALDriver
        
    #initializes a noised (with dropout) TF model
    def initializeNoisedTFModel(self):
        
        #get an uncompiled model with given parameters
        self.TFModel = get_noised_inceptionV3(dense_num_neurons = self.SSALDriver.numDenseNeurons, drop_rate = self.SSALDriver.dropoutRate, model_type = self.SSALDriver.modelType, num_classes = self.SSALDriver.numClasses)
        
        #compile model    
        self.compileTFModel(learning_rate = self.SSALDriver.LLLearningRate, label_smoothing = self.SSALDriver.labelSmoothing, optimizer = self.SSALDriver.optimizer, loss_func = self.SSALDriver.lossFunction)
        
    #initializes a denoised (no dropout) TF model
    def initializeDeNoisedTFModel(self):
        
        #get an uncompiled model with given parameters
        self.TFModel = get_denoised_inceptionV3(dense_num_neurons = self.SSALDriver.numDenseNeurons,)
        
        #compile model    
        self.compileTFModel(learning_rate = self.SSALDriver.LLLearningRate, label_smoothing = self.SSALDriver.labelSmoothing, optimizer = self.SSALDriver.optimizer, model_type = self.SSALDriver.modelType, num_classes = self.SSALDriver.numClasses, loss_func = self.SSALDriver.lossFunction)
        
        
    #compile the model with different parameters
    def compileTFModel(self, label_smoothing = 0, learning_rate = 0.0001, optimizer = 'adam', loss_func = "CCE"):
        
        self.TFModel = compile_model(self.TFModel, learning_rate = learning_rate, label_smoothing = label_smoothing, optimizer = optimizer, loss_func = loss_func)
        
        print(self.TFModel.summary())
        
     
    #train the TF model, uses current driver training set
    def trainTFModel(self, num_epochs = 1, batch_size = 32):
 
        #facilitates early stopping when categorical accuracy doesn't improve for a specified number of iterations
        if(self.SSALDriver.earlyStoppingPatience == -1):
            
            es = None
        else:
            
            es = EarlyStopping(monitor = "val_categorical_accuracy", mode = "max", verbose = 1, patience = self.SSALDriver.earlyStoppingPatience)
            
 
        #get the number of images in training set to take correctly sized steps
        num_train_images = self.SSALDriver.currentTrainingSet.shape[0]
        
        #creates the tensorflow dataset
        ds_train = construct_tf_train_dataset(self.SSALDriver.currentTrainingSet,batch_size = self.SSALDriver.batchSize, augment = self.SSALDriver.augmentImages, num_classes = self.SSALDriver.numClasses)
    
        #constructs validation dataset if specified
        if(self.SSALDriver.useValidation == True):
            
            ds_valid = construct_tf_validation_dataset(self.SSALDriver.validationSetDF, self.SSALDriver.batchSize, self.SSALDriver.numClasses)
        else:
            ds_valid = None
        
        print('optimizer')
        print(self.TFModel.optimizer.get_config())
        
        
        print('loss')
        print(self.TFModel.loss.get_config())
        print()
    
        #train the last layers
        history_ll = self.TFModel.fit(ds_train, callbacks = es,validation_data = ds_valid, shuffle = True, epochs = self.SSALDriver.numLLEpochs, batch_size = self.SSALDriver.batchSize, class_weight = self.SSALDriver.classWeightsDict, steps_per_epoch = math.ceil(num_train_images/self.SSALDriver.batchSize), verbose = 1)
        
        #save history
        history_df = pd.DataFrame(history_ll.history)
        history_df["train_type"] = "last_layer"
    

        #fine tune the whole model
        if(self.SSALDriver.fineTuneEpochs > 0):
            
            for layer in self.TFModel.layers:
                layer.trainable = True
                
            self.compileTFModel(self.SSALDriver.fineTuneLearningRate, self.SSALDriver.labelSmoothing)
                
                
            history_ft = self.TFModel.fit(ds_train, callbacks = es, validation_data = ds_valid, shuffle = True, epochs = self.SSALDriver.fineTuneEpochs, batch_size = self.SSALDriver.batchSize, class_weight = self.SSALDriver.classWeightsDict, steps_per_epoch = math.ceil(num_train_images/self.SSALDriver.batchSize), verbose = 1)
            
            #save history
            history_ft_df = pd.DataFrame(history_ft.history)
            history_ft_df["train_type"] = "fine_tune"
            
            history_df = history_df.append(history_ft_df, ignore_index = True)
            
           
            
            
        history_df.to_csv(self.SSALDriver.saveDirectory + "iteration_histories/" + self.SSALDriver.name + "_history_" + str(self.SSALDriver.onIteration) + ".csv")
        
        
      
            
     
    #make predictions given the current model and driver prediction set    
    def predictTFModel_prediction_set(self, batch_size = 32):
        
        #get the number of images in prediction set to take correctly sized steps
        num_predict_images = self.SSALDriver.currentPredictionSet.shape[0]
        
        #creates the tensorflow prediction dataset
        ds_pred = construct_tf_predict_dataset(self.SSALDriver.currentPredictionSet, batch_size = self.SSALDriver.batchSize)
        
        #gets the name of the images from the dataset to build full prediction dataframe
        if(num_predict_images != 0):
            image_names = np.concatenate([z.numpy() for x,y,z, in ds_pred], axis = 0)
            image_names = image_names.astype("U13")
        else:
            image_names = []
        
        #make predictions
        time_before = time.time()
        all_predictions = self.TFModel.predict(ds_pred)
        time_after = time.time()
        
        time_taken = time_after - time_before
        
        print("time for predictions",time_taken)
        
        #calculate single prediction
        single_predictions = np.argmax(all_predictions, axis = 1)
        
        #throw probabilities into a dataframe
        probs = get_prob_list(num_classes = self.SSALDriver.numClasses)
        
        predictions_df = pd.DataFrame(all_predictions, columns = probs)
       
        #for calculations later
        predictions_df["predicted_label"] = single_predictions
        predictions_df["citizen_label"] = 0
        predictions_df["image_name"] = image_names
        
        
        #assign predictions to driver data member        
        self.SSALDriver.predictionResults = predictions_df
        
        #add citizen labels to predictionResults
        self.SSALDriver.add_citizen_labels_to_predictions()
        
        self.SSALDriver.add_dataset_labels_to_predictions()
    
    #make evaluation predictions given the current model and evaluation set    
    def predictTFModel_evaluation_set(self, batch_size = 32):
        
        #creates the tensorflow prediction dataset
        ds_eval = construct_tf_evaluation_dataset(self.SSALDriver.evaluationSetDF, batch_size = self.SSALDriver.batchSize)
        
        #gets the name of the images from the dataset to build full prediction dataframe
        image_names = np.concatenate([z.numpy() for x,y,z, in ds_eval], axis = 0)
        image_names = image_names.astype("U13")
        
        official_labels = np.concatenate([y.numpy() for x,y,z, in ds_eval], axis = 0)
        
        #make predictions
        all_predictions = self.TFModel.predict(ds_eval)
        
        #calculate single prediction
        single_predictions = np.argmax(all_predictions, axis = 1)
        
        probs = get_prob_list(num_classes = self.SSALDriver.numClasses)
        
        #throw probabilities into a dataframe
        predictions_df = pd.DataFrame(all_predictions, columns = probs)
       
        #for calculations later
        predictions_df["citizen_label"] = np.nan
        predictions_df["predicted_label"] = single_predictions
        predictions_df["official_label"] = official_labels
        predictions_df["image_name"] = image_names
        
        
        self.SSALDriver.evaluationPredictionResults = predictions_df
        


