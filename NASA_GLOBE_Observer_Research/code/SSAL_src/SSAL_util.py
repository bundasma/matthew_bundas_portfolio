import numpy as np

def get_label_list(num_classes, base_term = "acc"):
    
    base = "_" + str(base_term)
    acc_list = ["total" + base]
    
    for i in range(num_classes):
        
        acc_i = str(i) + base
        acc_list.append(acc_i)
        
    return acc_list

def get_prob_list(num_classes = 10):
       
       base = "_prob"
       prob_list = []
       
       for i in range(num_classes):
           
           prob_i = str(i) + base
           prob_list.append(prob_i)
           
       return prob_list


def get_dataset_label_names(dataset = "GLOBE", num_classes = 10):
    
    acceptable_datasets = ["GLOBE","intel"]
    
    
    if(dataset == "GLOBE"):
        
        return ['total','0 - Closed Forest','1 - Woodland', "2 - Shrub/Thic", "3 - Dwarf Shrub/Thic", "4 - Herbaceous", "5 - Barren", "6 - Wetland", "7 - Open Water", "8 - Cult. Land", "9 - Urban", "10 - Bad Image"][:(num_classes + 1)]
    elif(dataset == "intel"):
        
        return ['total','0 - Buildings','1 - Forest', "2 - Glacier", "3 - Mountain", "4 - Sea", "5 - Street"][:(num_classes + 1)]
    else:
        print("invalid dataset choose:")
        print(acceptable_datasets)  
        
        print("outputting default")
        
        default_names = ["total"]
        
        default_numbers = np.arange(6).astype(str)
        
        default_names = default_names.append(default_numbers)
        
        return default_names
        
        


def calculate_correct_end_population(current_distribution, num_confident, num_classes = 10):
    
    limiting_class_i = np.argmin(num_confident / current_distribution)
    
    limiting_class_value = num_confident[limiting_class_i]
    
    current_distribution_per = current_distribution/np.sum(current_distribution)
    limiting_class_ratio = current_distribution_per[limiting_class_i]


    prop = np.zeros(num_classes)

    for i in range(len(prop)):

        if(i == limiting_class_i):

            prop[i] = limiting_class_value

        else:

            factor = current_distribution_per[i] / limiting_class_ratio

            prop[i] = limiting_class_value * factor
            
    prop = np.ceil(prop)    
    return prop.astype(int)

def mod_ent_df_func(row, distribution, num_classes = 10):
    
    probs_list = get_prob_list(num_classes)
    probabilities = row[probs_list]
    
    return calc_modified_entropy(probabilities, distribution)

def reg_ent_df_func(row, num_classes = 10):

    probs_list = get_prob_list(num_classes)
    probabilities = row[probs_list]

    return calc_regular_entropy(probabilities)

def calc_modified_entropy(probs, distribution):
    
    entropy = 0

    print(distribution)

    for i in range(len(probs)):
        
        entropy += probs[i] * np.log(probs[i]) * np.abs(np.log(distribution[i]))**2
        
    return -entropy

def calc_regular_entropy(probs):
    
    entropy = 0
    
    for i in range(len(probs)):
        
        entropy += probs[i] * np.log(probs[i])
        
    return -entropy

def correctly_citizen_labeled(row):
        
        actual_label = row["acutal_label"]
        citizen_label = row["citizen_label"]
 
        if(actual_label == citizen_label):
            
            return 1
    
        else:
            return 0