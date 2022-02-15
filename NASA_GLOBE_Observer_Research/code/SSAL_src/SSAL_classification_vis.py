from SSAL_src.SSAL_evaluation import load_labeled_data, load_SSAL_predictions
import matplotlib.pyplot as plt
from skimage.transform import resize
from IPython.display import clear_output
import pandas as pd
import numpy as np


#determines if the row is misclassified
def misclassified_func(x):
    
    if(x["official_label"] == x["prediction"]):
        
        return 0
    
    else:
        
        return 1

#load filtered portions of data to look at how well citizens labeled
def load_citizen_predictions_for_error_vis(label_type = "incorrect"):
    
    labeled_data = load_labeled_data()
    
    #rename them to align with model outputs, just to make it easier, uniform
    labeled_data = labeled_data.rename(columns = {'citizen_label':'prediction'})
    
    
    labeled_data["misclassified"] = labeled_data.apply(misclassified_func, axis = 1)
    misclassified = labeled_data[labeled_data["misclassified"] == 1]
    corrclassified = labeled_data[labeled_data["misclassified"] == 0]

    if(label_type == "incorrect"):
        print("returning misclassified")
        return misclassified
    elif(label_type == "correct"):
        
        print("returning corrclassified")
        return corrclassified
    else:
        print("returning all")
        return labeled_data
        
        

#load model predictions, either incorrect, correct, or any prediction
def load_model_predictions_for_error_vis(model_name,label_type = "incorrect"):
    
    full_pool = load_SSAL_predictions(model_name)
    full_pool = full_pool.rename(columns = {'predicted_label':'prediction'})
    full_pool["misclassified"] = full_pool.apply(misclassified_func, axis = 1)
    
    
    if(label_type == "incorrect"):
        
        return full_pool[full_pool["misclassified"] == 1]
    
    elif(label_type == "correct"):
        return full_pool[full_pool["misclassified"] == 0]

    else:
        return full_pool
    

    

    
#function to handle user input
def userInput():
    
    user_input = input()
    
    return user_input
    
    
def visualize_classifications(examine_model, classification_type, show_class, show_probabilities = False):
    
    user_input = "start"

    #load from predictions or portioned data
    if(examine_model == "citizen_scientists"):

        image_pool = load_citizen_predictions_for_error_vis(label_type = classification_type)
        
        #path to images themselves
        image_path = "./data/images/all_images/"

    else:

        image_pool = load_model_predictions_for_error_vis(examine_model, label_type = classification_type)
        image_path = "./data/images/all_images/"


    #restrict class if needed
    if(show_class != "all"):

        image_pool = image_pool[image_pool["official_label"] == show_class]

    #keep going until user enters stop
    while(user_input != "stop"):

        
        if(image_pool.shape[0] >= 9):
            replace = False
        elif(image_pool.shape[0] == 0):
            print("no images meet criteria")
            return
            
        else:
            replace = True
        
        #get 9 random images
        nine_sample = image_pool.sample(n=9, replace = replace)

        #initiate subplots,axes
        _, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.flatten()

     


        #keeps track of which image we are on 0-8
        i = 0

        #plot one image per ax
        for ax in axs:

            #get data row
            row = nine_sample.iloc[i]

            #get image and labels/predictions
            image_name = row["image_name"]
            prediction = row["prediction"]
            actual_label = row["official_label"]
            
         
            
            

            #load and resize image
            try:
                img = plt.imread(image_path + image_name)
                #img = plt.imread(image_name)
            except:
                print("couldnt load",image_name)
                continue
            img = resize(img, (300,300))    


            ax.imshow(img)
            ax.set_title(str(actual_label) + " predicted as " + str(prediction))
            #ax.title.set_text(str(probabilities))
            
           
            if(show_probabilities == True and examine_model != "citizen_scientists"):
               
                probabilities = row.iloc[1:11].values
            
                probabilities_rounded = np.round(probabilities.astype(np.double),2)
            
                probabilities = list(probabilities_rounded)[:-1]
                ax.set_xlabel(image_name + "\n" + str(probabilities), labelpad = -25, color = "white", fontsize = 7.5)
                
            else:
                ax.set_xlabel(image_name, labelpad = -20, color = "white")
                
                
            #ax.set_ylabel(str(probabilities),fontsize = 8)
            
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])
           # ax.axis("off")
           
            



            i+=1

        #plot key
        plt.plot([],[],label = "Label Key\n0 - Closed Forest\n1 - Woodland\n2 - Shrubland/Thicket\n3 - Dwarf Shrubland\n4 - Herbaceous Veg\n5 - Barren\n6 - Wetland\n7 - Open Water\n8 - Cultivated Land\n9 - Urban \n10 - Bad Image\n\nstop to stop loop", color = "white")
        plt.legend(loc = 1, bbox_to_anchor = (2.0,3.40), fontsize = 12)    

        plt.show()

        #get user input
        user_input = userInput()


        #clear output for next image
        if(user_input != "stop"):
            clear_output(wait = False)
            
            
def visualize_multiimage_classifications(data, show_class = "all", show_probabilities = False):
    
    
    title_dict = {0:"0 - Closed Forest", 1:"1 - Woodland", 2:"2 - Shrubland/Thicket", 3:"3 - Dwarf Shrubland", 4:"4 - Herbaceous Veg", 5:"5 - Barren", 6:"6 - Wetland", 7:"7 - Open Water", 8:"8 - Cultivated Land", 9:"9 - Urban", 10:"10 - Bad Image"}
    

    user_input = "start"

    #restrict class if needed
    if(show_class != "all"):

        image_pool = image_pool[image_pool["actual_label"] == show_class]

    #keep going until user enters stop
    while(user_input != "stop"):

        random_image_ind = np.random.randint(data.shape[0])
        #get 9 random images
        random_row = data.iloc[random_image_ind]
        
        row_label = random_row["label"]

        #initiate subplots,axes
        _, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        main_title = title_dict[int(row_label)]
        _.suptitle(main_title, y = 0.90, fontsize = 16)


        #keeps track of which image we are on 0-8
        i = 0

        #plot one image per ax
        for ax in axs:

            #get image name
            image = random_row[i]

            
            #load and resize image
            try:
                img = plt.imread(image)
                #img = plt.imread(image_name)
            except:
                print("couldnt load",image)
                continue
            img = resize(img, (300,300))    


            ax.imshow(img)
    
        
            
            ax.set_xlabel(image, labelpad = -20, color = "white")
                
                
            #ax.set_ylabel(str(probabilities),fontsize = 8)
            
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])
           # ax.axis("off")

            i+=1

        #plot key
        plt.plot([],[],label = "Label Key\n0 - Closed Forest\n1 - Woodland\n2 - Shrubland/Thicket\n3 - Dwarf Shrubland\n4 - Herbaceous Veg\n5 - Barren\n6 - Wetland\n7 - Open Water\n8 - Cultivated Land\n9 - Urban\n\nstop to stop loop", color = "white")
        plt.legend(loc = 1, bbox_to_anchor = (1.65,2.25), fontsize = 12)    

        plt.show()

        #get user input
        user_input = userInput()


        #clear output for next image
        if(user_input != "stop"):
            clear_output(wait = False)