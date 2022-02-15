print("importing in SSAL_models")

import keras
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from keras import optimizers

from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121

import numpy as np

import tensorflow as tf
import pandas as pd
import os

import time
import math

print("done importing in SSAL_models")


#def box_cox_loss()


#creates and returns an uncompiled InceptionV3 model
def get_noised_inceptionV3(dense_num_neurons = 1024, drop_rate = 0.5, model_type = "inception", num_classes = 10):
    
    #initialize model
    if(model_type == "inception"):
        model = InceptionV3(include_top = False, input_shape = (299,299,3))
    elif(model_type == "densenet"):
        model =  DenseNet121(include_top = False, weights = "imagenet", input_shape = (299,299,3), pooling = "avg")
    else:
        print("bad model type")
        return
    
    #do not train intermediate feature extraction layers
    for layer in model.layers:
        
        layer.trainable = False
    
    #flatten and add dense and classification layer
    flattened = Flatten()(model.layers[-1].output)
    dropout = Dropout(rate = drop_rate)(flattened)
    dense = Dense(dense_num_neurons, activation = "relu")(dropout)
    classification = Dense(num_classes, activation = "softmax")(dense)
    
    complete_model = Model(inputs = model.inputs, outputs = classification)
    
    #loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing)
    #optimizer = optimizers.Adam(learning_rate = learning_rate)
    #metrics = ["categorical_accuracy"]
    
    
    #complete_model.compile(loss = loss_func, optimizer = optimizer, metrics = metrics)
    
    return complete_model


#creates and returns an uncompiled InceptionV3 model
def get_denoised_inceptionV3(dense_num_neurons = 1024, model_type = 'inception', num_classes = 10):
    
   
    #initialize model
    if(model_type == "inception"):
        model = InceptionV3(include_top = False, input_shape = (299,299,3))
    elif(model_type == "densenet"):
        model =  DenseNet121(include_top = False, weights = "imagenet", input_shape = (299,299,3), pooling = "avg")
    else:
        print("bad model type")
        return
    
        
 
    
    #flatten and add dense and classification layer
    flattened = Flatten()(model.layers[-1].output)
    
    dense = Dense(dense_num_neurons, activation = "relu")(flattened)
    
    classification = Dense(num_classes, activation = "softmax")(dense)
    
    
    complete_model = Model(inputs = model.inputs, outputs = classification)
    
    
    #loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing)
    #optimizer = optimizers.Adam(learning_rate = learning_rate)
    #metrics = ["categorical_accuracy"]
    
    
    #complete_model.compile(loss = loss_func, optimizer = optimizer, metrics = metrics)
    
    return complete_model

def compile_model(model, label_smoothing = 0, learning_rate = 0.0001, optimizer = "adam", loss_func = "CCE"):
    
    #original
    if(loss_func == "CCE"):
        loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing)
    elif(loss_func == "MAE"):
        loss_func = tf.keras.losses.MeanAbsoluteError()
    else:
        print("invalid loss function, using CCE")
        loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing)
    
        
    #optimizer = optimizers.Adam(learning_rate = learning_rate)
    
    #internet's
    #loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    #optimizer = keras.optimizers.SGD(lr = learning_rate)
    
    
    
    if(optimizer == "adam"):
        optimizer = optimizers.Adam(learning_rate = learning_rate)
    elif(optimizer == "sgd"):
        optimizer = optimizers.SGD(learning_rate = learning_rate)
    else:
        print("bad optimizer")
        return
        
        
   
    metrics = ["categorical_accuracy"]
    
    
    model.compile(loss = loss_func, optimizer = optimizer, metrics = metrics)
    

    
    return model

#reads in an image, usually for a training dataset
def read_image(image_file, label):

    #source directory of the images
    image_dir = "./data/images/all_images/"    

    image = tf.io.read_file(image_dir + image_file)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.image.resize(image,[299,299])


    return image/255.0, label

#reads a prediction image, label doesn't matter, returns image name
def read_image_pred(image_name, label = -1):

    #source directory of the images
    image_dir = "./data/images/all_images/"
    
    image = tf.io.read_file(image_dir + image_name)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.image.resize(image,[299,299])


    return image/255.0, label, image_name

#reads an evaluation image
def read_image_eval(image_name, label = -1):

    #source directory of the images
    image_dir = "./data/images/all_images/"
    
    image = tf.io.read_file(image_dir + image_name)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.image.resize(image,[299,299])


    return image/255.0, label, image_name

#configure a training dataset
def config_performance(ds, batch_size):

    #batch the dataset
    ds = ds.batch(batch_size, drop_remainder = True)
    
    #repeat the dataset, make it infinite
    ds = ds.repeat()
    
    #prefetch, fill the buffer
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

#configure a prediction dataset
def config_performance_pred(ds, batch_size):
    
    #batch the dataset
    ds = ds.batch(batch_size)
    
    #prefetch, fill the buffer
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

#configure an evaluation dataset
def config_performance_eval(ds, batch_size):
    
    #batch the dataset
    ds = ds.batch(batch_size)
    
    #prefetch, fill the buffer
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds
    
    

#handles turning the training dataframe into a tensorflow dataset
def construct_tf_train_dataset(train_df, batch_size = 32, augment = True, num_classes = 10):
    
    #get x and y
    train_x = train_df["image_name"]
    train_y = train_df["official_label"]
    train_y = keras.utils.to_categorical(train_y, num_classes)
    
    
    num_train_images = np.shape(train_x)[0]
    
    #create tf.dataset, shuffle, map into infinite dataset
    ds_train = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds_train = ds_train.shuffle(buffer_size = num_train_images)
    ds_train = ds_train.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    if(augment == True):
        
        #augment the data
        ds_train = ds_train.map( lambda x, y: tf.py_function(augment_img, [x,y], [tf.float32, tf.float32]), num_parallel_calls = tf.data.experimental.AUTOTUNE)

        img_shape = (299,299,3)
        #force to be the correct shapes, otherwise there's a bug
        ds_train = ds_train.map(lambda img, label: set_shapes(img,label,img_shape, num_classes))

    #turn it into an infinite dataset with batches
    ds_train = config_performance(ds_train,batch_size)
    
    return ds_train


    
#handles turning a dataframe into a dataset for in iteration predictions
def construct_tf_predict_dataset(predict_df, batch_size = 32, num_classes = 10):
    
    #get x and y
    predict_x = predict_df["image_name"]
    
    #create tf.dataset, shuffle, map into infinite dataset
    ds_predict = tf.data.Dataset.from_tensor_slices((predict_x))
    
    ds_predict = ds_predict.map(read_image_pred, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #turn it into an infinite dataset with batches
    ds_predict = config_performance_pred(ds_predict,batch_size)
    
    return ds_predict   

#handles turning a dataframe into a dataset for evaluation
def construct_tf_validation_dataset(validation_df, batch_size = 32, num_classes = 10):
    
    #get x and y
    valid_x = validation_df["image_name"]
    valid_y = validation_df["official_label"]
    
    
    print(valid_y)
    valid_y = keras.utils.to_categorical(valid_y, num_classes)
    
    #create tf.dataset, shuffle, map into infinite dataset
    ds_valid = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    
    ds_valid = ds_valid.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #turn it into an infinite dataset with batches
    ds_valid = config_performance_eval(ds_valid,batch_size)
    
    return ds_valid

#handles turning a dataframe into a dataset for evaluation
def construct_tf_evaluation_dataset(evaluate_df, batch_size = 32):
    
    #get x and y
    evaluate_x = evaluate_df["image_name"]
    evaluate_y = evaluate_df["official_label"]
    
    #create tf.dataset, shuffle, map into infinite dataset
    ds_evaluate = tf.data.Dataset.from_tensor_slices((evaluate_x, evaluate_y))
    
    ds_evaluate = ds_evaluate.map(read_image_eval, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #turn it into an infinite dataset with batches
    ds_evaluate = config_performance_eval(ds_evaluate,batch_size)
    
    return ds_evaluate



def random_brightness(image):

    return tf.image.random_brightness(image,0.1)

def random_saturation(image):

    return tf.image.random_saturation(image, 0.5, 1.5)

def random_contrast(image):

    return tf.image.random_contrast(image, 0.5, 1.5)

def random_shear(image):

    if(type(image) != np.ndarray):

        image = image.numpy()

    return tf.keras.preprocessing.image.random_shear(image,10, fill_mode = "nearest", channel_axis = 2, row_axis = 1, col_axis = 0)

def random_rotation(image):

    if(type(image) != np.ndarray):

        image = image.numpy()

    return tf.keras.preprocessing.image.random_rotation(image,20, fill_mode = "nearest", channel_axis = 2, row_axis = 1, col_axis = 0)


def random_shift(image):

    if(type(image) != np.ndarray):

        image = image.numpy()

    return tf.keras.preprocessing.image.random_shift(image, 0.08, 0.08,  fill_mode = "nearest", channel_axis = 2, row_axis = 1, col_axis = 0)


def random_zoom(image):

    if(type(image) != np.ndarray):

        image = image.numpy()


    return tf.keras.preprocessing.image.random_zoom(image, (0.88, 0.88),  fill_mode = "nearest", channel_axis = 2, row_axis = 1, col_axis = 0)


all_augmentations = [random_brightness, random_saturation, random_contrast, random_shear, random_rotation, random_shift, random_zoom]


def augment_img(image,label):

    number_augs = np.random.randint(8)
   
    do_augmentations = np.random.choice(all_augmentations,number_augs, replace = False)


    for aug in do_augmentations:

        image = aug(image)


    image = tf.image.random_flip_left_right(image)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

#force dataset to be the correct shape to avoid tensorflow bug when using custom map function
def set_shapes(image, label, shape, num_classes = 10):

    image.set_shape(shape)
    label.set_shape([num_classes])
    return image, label