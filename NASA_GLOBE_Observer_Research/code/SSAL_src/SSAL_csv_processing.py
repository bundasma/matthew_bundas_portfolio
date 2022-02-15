import pandas as pd
import numpy as np
import random
import os.path
from os import path
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copyfile
import sys
import glob



#checks if the image exists (if we have it downloaded)
def image_exists(image_name):

    image_dir = "./data/images/all_images/"

    if(path.exists(image_dir + image_name)):

        return 1

    else:

        return 0
    
#legacy loading and processing raw data csv file    
def load_process_csv_old(csvpath = "./data/raw_data.csv"):

    #load the data
    raw_data = pd.read_csv(csvpath)

    #get rid of the description row
    raw_data.drop(raw_data.index[0], inplace = True)

    #have the dataframe index start from 0
    raw_data.reset_index(inplace = True)
    raw_data.drop(columns = "index", inplace = True)


    #columns we want to keep and rename
    raw_data_keep_cols = ["organization_id","land covers:userid","measured_on","site_id","land covers:land cover id","land covers:muc code","land covers:muc description", "land covers:north photo url","land covers:north classifications","land covers:east photo url","land covers:east classifications","land covers:south photo url","land covers:south classifications","land covers:west photo url","land covers:west classifications"]

    #new names of columns
    raw_data_col_names = ["org_id","user_id","measured_on","site_id","cover_id","muc_code","cover_desc","north_url","north_class","east_url","east_class","south_url","south_class","west_url","west_class"]

    #columns we want to consider when dropping duplicates, ie if all urls are the same, consider dropping
    dup_drop_cols = ["north_url","east_url","south_url","west_url"]

    #change column names
    raw_data = raw_data[raw_data_keep_cols]
    raw_data.columns = raw_data_col_names

    total_rows = raw_data.shape[0]

    #drop rows where they are duplicates except for cover_id
    raw_data.drop_duplicates(subset = dup_drop_cols, inplace = True, keep = "last")

    total_rows_after_dup = raw_data.shape[0]

    #force cover_id to be an integer
    raw_data["cover_id"] = raw_data["cover_id"].astype(int)
    #raw_data["org_id"] = raw_data["org_id"].astype(int)
    #raw_data["user_id"] = raw_data["user_id"].astype(int)

    print("Number of raw rows,",total_rows)
    print("Number of total rows after dropping duplicates,",total_rows_after_dup)

    return raw_data
def load_process_csv(csvpath = "./data/raw_data.csv"):

    #load the data
    raw_data = pd.read_csv(csvpath)

    #get rid of the description row
    raw_data.drop(raw_data.index[0], inplace = True)

    #have the dataframe index start from 0
    raw_data.reset_index(inplace = True)
    raw_data.drop(columns = "index", inplace = True)


    #columns we want to keep and rename
    raw_data_keep_cols = ["organization_id","land covers:userid","measured_on","site_id","land covers:land cover id","land covers:muc code","land covers:muc description", "land covers:north photo url","land covers:north classifications","land covers:east photo url","land covers:east classifications","land covers:south photo url","land covers:south classifications","land covers:west photo url","land covers:west classifications"]

    #new names of columns
    raw_data_col_names = ["org_id","user_id","measured_on","site_id","cover_id","muc_code","cover_desc","n_url","n_class","e_url","e_class","s_url","s_class","w_url","w_class"]

    #columns we want to consider when dropping duplicates, ie if all urls are the same, consider dropping
    dup_drop_cols = ["n_url","e_url","s_url","w_url"]

    #change column names
    raw_data = raw_data[raw_data_keep_cols]
    raw_data.columns = raw_data_col_names

    total_rows = raw_data.shape[0]

    #drop rows where they are duplicates except for cover_id
    raw_data.drop_duplicates(subset = dup_drop_cols, inplace = True, keep = "last")

    total_rows_after_dup = raw_data.shape[0]

    #force cover_id to be an integer
    raw_data["cover_id"] = raw_data["cover_id"].astype(int)
    #raw_data["org_id"] = raw_data["org_id"].astype(int)
    #raw_data["user_id"] = raw_data["user_id"].astype(int)

    #print("Number of raw rows,",total_rows)
    #print("Number of total rows after dropping duplicates,",total_rows_after_dup)


    #separate directions
    norths = raw_data[["cover_id","n_url","n_class"]].rename(columns = {"n_url":"url","n_class":"class"})
    easts = raw_data[["cover_id","e_url","e_class"]].rename(columns = {"e_url":"url","e_class":"class"})
    souths = raw_data[["cover_id","s_url","s_class"]].rename(columns = {"s_url":"url","s_class":"class"})
    wests = raw_data[["cover_id","w_url","w_class"]].rename(columns = {"w_url":"url","w_class":"class"})

    norths["direction"] = "n"
    easts["direction"] = "e"
    souths["direction"] = "s"
    wests["direction"] = "w"

    #creates the image_name column for each direction
    norths["image_name"] =  norths["cover_id"].astype(str) + "_n.jpg"
    easts["image_name"] =  easts["cover_id"].astype(str) + "_e.jpg"
    souths["image_name"] = souths["cover_id"].astype(str) + "_s.jpg"
    wests["image_name"] =  wests["cover_id"].astype(str) + "_w.jpg"

    #creates the label column for each direction by applying calculate_primary_class function
    norths["citizen_label"] = norths["class"].apply(calculate_primary_class).astype(pd.Int64Dtype())
    easts["citizen_label"] = easts["class"].apply(calculate_primary_class).astype(pd.Int64Dtype())
    souths["citizen_label"] = souths["class"].apply(calculate_primary_class).astype(pd.Int64Dtype())
    wests["citizen_label"] = wests["class"].apply(calculate_primary_class).astype(pd.Int64Dtype())

    #concatinate/combine the directions
    baseline_data = pd.concat([norths,easts,souths,wests])

    baseline_data["downloaded"] = baseline_data["image_name"].apply(image_exists)

    baseline_data = baseline_data[baseline_data["downloaded"] == 1]

    baseline_data = baseline_data.sample(frac = 1)

    baseline_data["official_label"] = np.nan
    baseline_data["official_label_source"] = np.nan

    return baseline_data[["image_name","url","citizen_label","official_label","official_label_source"]]


def load_web_data(web_data_path = "./data/web_raw_data.csv"):

    web_data = pd.read_csv(web_data_path, index_col = 0)

    web_data["image_name"] = web_data["web_id"].astype(str) + "_web.jpg"
    web_data["citizen_label"] = np.nan
    web_data["official_label_source"] = "web"

    web_data = web_data.rename(columns = {"actual_label":"official_label","URL":"url"})

    return web_data[["image_name","url","citizen_label","official_label","official_label_source"]]


#inputs a raw class string like "90% MUC 93 [Urban, Roads and Parking]; 30% MUC 91 [Urban, Residential Property]"
#outputs the integer muc code class, handle applying it later
def calculate_primary_class(raw_class):

    if(raw_class is np.nan):

        return np.nan

    sep = raw_class.split(';')
    #print(sep)
    final = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in sep:

        try:
            mucPos = i.index('MUC')
        except:
            mucPos = i.index('M')


        mucNum = int(i[mucPos + 4])

        if (i[0] == ' '):
            start = 1
        else:
            start = 0

        currPos = start
        while (i[currPos] != '%' or currPos > len(i)):
            currPos += 1

        #print(i[start])
        #print('MN: ' + str(mucNum))
        #print('percentage: ' + i[start:currPos])
        final[mucNum] += int(i[start:currPos])
        #print(final)


    #get indexes of maxes
    maxes = np.argwhere(final == np.amax(final))

    #if there is one max, pick that index
    if(len(maxes) == 1):

        dominateClass = int(maxes[0])

    #if there is not one max, pick a random index
    else:

        dominateClass = int(random.choice(maxes))



    if dominateClass == -1:
        print('Something went horribly wrong')
        quit()

    return dominateClass
