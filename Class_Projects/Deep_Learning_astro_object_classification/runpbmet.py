import pandas as pd
import numpy as np


def dopbstuff(measurements):

    obj_id_group = measurements.groupby("object_id", as_index = False).agg({"flux":["mean"]})


    grouped = measurements.groupby(['object_id','passband'], as_index = False).agg({"flux":["max","mean","median","min",np.std]})
    grouped.columns = list(map(''.join, grouped.columns.values))


    obj_id_group["std0"] = np.asarray(grouped[grouped["passband"] == 0]["fluxstd"])
    obj_id_group["std1"] = np.asarray(grouped[grouped["passband"] == 1]["fluxstd"])
    obj_id_group["std2"] = np.asarray(grouped[grouped["passband"] == 2]["fluxstd"])
    obj_id_group["std3"] = np.asarray(grouped[grouped["passband"] == 3]["fluxstd"])
    obj_id_group["std4"] = np.asarray(grouped[grouped["passband"] == 4]["fluxstd"])
    obj_id_group["std5"] = np.asarray(grouped[grouped["passband"] == 5]["fluxstd"])

    obj_id_group["median0"] = np.asarray(grouped[grouped["passband"] == 0]["fluxmedian"])
    obj_id_group["median1"] = np.asarray(grouped[grouped["passband"] == 1]["fluxmedian"])
    obj_id_group["median2"] = np.asarray(grouped[grouped["passband"] == 2]["fluxmedian"])
    obj_id_group["median3"] = np.asarray(grouped[grouped["passband"] == 3]["fluxmedian"])
    obj_id_group["median4"] = np.asarray(grouped[grouped["passband"] == 4]["fluxmedian"])
    obj_id_group["median5"] = np.asarray(grouped[grouped["passband"] == 5]["fluxmedian"])

    obj_id_group["mean0"] = np.asarray(grouped[grouped["passband"] == 0]["fluxmean"])
    obj_id_group["mean1"] = np.asarray(grouped[grouped["passband"] == 1]["fluxmean"])
    obj_id_group["mean2"] = np.asarray(grouped[grouped["passband"] == 2]["fluxmean"])
    obj_id_group["mean3"] = np.asarray(grouped[grouped["passband"] == 3]["fluxmean"])
    obj_id_group["mean4"] = np.asarray(grouped[grouped["passband"] == 4]["fluxmean"])
    obj_id_group["mean5"] = np.asarray(grouped[grouped["passband"] == 5]["fluxmean"])

    obj_id_group["max0"] = np.asarray(grouped[grouped["passband"] == 0]["fluxmax"])
    obj_id_group["max1"] = np.asarray(grouped[grouped["passband"] == 1]["fluxmax"])
    obj_id_group["max2"] = np.asarray(grouped[grouped["passband"] == 2]["fluxmax"])
    obj_id_group["max3"] = np.asarray(grouped[grouped["passband"] == 3]["fluxmax"])
    obj_id_group["max4"] = np.asarray(grouped[grouped["passband"] == 4]["fluxmax"])
    obj_id_group["max5"] = np.asarray(grouped[grouped["passband"] == 5]["fluxmax"])

    obj_id_group["min0"] = np.asarray(grouped[grouped["passband"] == 0]["fluxmin"])
    obj_id_group["min1"] = np.asarray(grouped[grouped["passband"] == 1]["fluxmin"])
    obj_id_group["min2"] = np.asarray(grouped[grouped["passband"] == 2]["fluxmin"])
    obj_id_group["min3"] = np.asarray(grouped[grouped["passband"] == 3]["fluxmin"])
    obj_id_group["min4"] = np.asarray(grouped[grouped["passband"] == 4]["fluxmin"])
    obj_id_group["min5"] = np.asarray(grouped[grouped["passband"] == 5]["fluxmin"])

    obj_id_group["ratiomedian01"] = obj_id_group["median0"] / obj_id_group["median1"]
    obj_id_group["ratiomedian23"] = obj_id_group["median2"] / obj_id_group["median3"]
    obj_id_group["ratiomedian34"] = obj_id_group["median3"] / obj_id_group["median4"]
    obj_id_group["ratiomedian45"] = obj_id_group["median4"] / obj_id_group["median5"]
    obj_id_group["ratiomedian02"] = obj_id_group["median0"] / obj_id_group["median2"]
    obj_id_group["ratiomedian13"] = obj_id_group["median1"] / obj_id_group["median3"]
    obj_id_group["ratiomedian35"] = obj_id_group["median3"] / obj_id_group["median5"]
    obj_id_group["ratiomedian12"] = obj_id_group["median1"] / obj_id_group["median2"]
    obj_id_group["ratiomedian03"] = obj_id_group["median0"] / obj_id_group["median3"]
    obj_id_group["ratiomedian14"] = obj_id_group["median1"] / obj_id_group["median4"]
    obj_id_group["ratiomedian25"] = obj_id_group["median2"] / obj_id_group["median5"]
    obj_id_group["ratiomedian04"] = obj_id_group["median0"] / obj_id_group["median4"]
    obj_id_group["ratiomedian15"] = obj_id_group["median1"] / obj_id_group["median5"]

    obj_id_group["ratiomean01"] = obj_id_group["mean0"] / obj_id_group["mean1"]
    obj_id_group["ratiomean12"] = obj_id_group["mean1"] / obj_id_group["mean2"]
    obj_id_group["ratiomean23"] = obj_id_group["mean2"] / obj_id_group["mean3"]
    obj_id_group["ratiomean34"] = obj_id_group["mean3"] / obj_id_group["mean4"]
    obj_id_group["ratiomean45"] = obj_id_group["mean4"] / obj_id_group["mean5"]
    obj_id_group["ratiomean02"] = obj_id_group["mean0"] / obj_id_group["mean2"]
    obj_id_group["ratiomean13"] = obj_id_group["mean1"] / obj_id_group["mean3"]
    obj_id_group["ratiomean35"] = obj_id_group["mean3"] / obj_id_group["mean5"]
    obj_id_group["ratiomean03"] = obj_id_group["mean0"] / obj_id_group["mean3"]
    obj_id_group["ratiomean14"] = obj_id_group["mean1"] / obj_id_group["mean4"]
    obj_id_group["ratiomean25"] = obj_id_group["mean2"] / obj_id_group["mean5"]
    obj_id_group["ratiomean04"] = obj_id_group["mean0"] / obj_id_group["mean4"]
    obj_id_group["ratiomean15"] = obj_id_group["mean1"] / obj_id_group["mean5"]

    obj_id_group["ratiomax01"] = obj_id_group["max0"] / obj_id_group["max1"]
    obj_id_group["ratiomax12"] = obj_id_group["max1"] / obj_id_group["max2"]
    obj_id_group["ratiomax23"] = obj_id_group["max2"] / obj_id_group["max3"]
    obj_id_group["ratiomax34"] = obj_id_group["max3"] / obj_id_group["max4"]
    obj_id_group["ratiomax45"] = obj_id_group["max4"] / obj_id_group["max5"]
    obj_id_group["ratiomax02"] = obj_id_group["max0"] / obj_id_group["max2"]
    obj_id_group["ratiomax13"] = obj_id_group["max1"] / obj_id_group["max3"]
    obj_id_group["ratiomax35"] = obj_id_group["max3"] / obj_id_group["max5"]
    obj_id_group["ratiomax03"] = obj_id_group["max0"] / obj_id_group["max3"]
    obj_id_group["ratiomax14"] = obj_id_group["max1"] / obj_id_group["max4"]
    obj_id_group["ratiomax25"] = obj_id_group["max2"] / obj_id_group["max5"]
    obj_id_group["ratiomax04"] = obj_id_group["max0"] / obj_id_group["max4"]
    obj_id_group["ratiomax15"] = obj_id_group["max1"] / obj_id_group["max5"]

    obj_id_group["ratiostd01"] = obj_id_group["std0"] / obj_id_group["std1"]
    obj_id_group["ratiostd12"] = obj_id_group["std1"] / obj_id_group["std2"]
    obj_id_group["ratiostd23"] = obj_id_group["std2"] / obj_id_group["std3"]
    obj_id_group["ratiostd34"] = obj_id_group["std3"] / obj_id_group["std4"]
    obj_id_group["ratiostd45"] = obj_id_group["std4"] / obj_id_group["std5"]
    obj_id_group["ratiostd02"] = obj_id_group["std0"] / obj_id_group["std2"]
    obj_id_group["ratiostd13"] = obj_id_group["std1"] / obj_id_group["std3"]
    obj_id_group["ratiostd35"] = obj_id_group["std3"] / obj_id_group["std5"]
    obj_id_group["ratiostd03"] = obj_id_group["std0"] / obj_id_group["std3"]
    obj_id_group["ratiostd14"] = obj_id_group["std1"] / obj_id_group["std4"]
    obj_id_group["ratiostd25"] = obj_id_group["std2"] / obj_id_group["std5"]
    obj_id_group["ratiostd04"] = obj_id_group["std0"] / obj_id_group["std4"]
    obj_id_group["ratiostd15"] = obj_id_group["std1"] / obj_id_group["std5"]

    obj_id_group["ratiomin01"] = obj_id_group["min0"] / obj_id_group["min1"]
    obj_id_group["ratiomin12"] = obj_id_group["min1"] / obj_id_group["min2"]
    obj_id_group["ratiomin23"] = obj_id_group["min2"] / obj_id_group["min3"]
    obj_id_group["ratiomin34"] = obj_id_group["min3"] / obj_id_group["min4"]
    obj_id_group["ratiomin45"] = obj_id_group["min4"] / obj_id_group["min5"]
    obj_id_group["ratiomin02"] = obj_id_group["min0"] / obj_id_group["min2"]
    obj_id_group["ratiomin13"] = obj_id_group["min1"] / obj_id_group["min3"]
    obj_id_group["ratiomin35"] = obj_id_group["min3"] / obj_id_group["min5"]
    obj_id_group["ratiomin03"] = obj_id_group["min0"] / obj_id_group["min3"]
    obj_id_group["ratiomin14"] = obj_id_group["min1"] / obj_id_group["min4"]
    obj_id_group["ratiomin25"] = obj_id_group["min2"] / obj_id_group["min5"]
    obj_id_group["ratiomin04"] = obj_id_group["min0"] / obj_id_group["min4"]
    obj_id_group["ratiomin15"] = obj_id_group["min1"] / obj_id_group["min5"]

    obj_id_group["diffmedian01"] = obj_id_group["median0"] - obj_id_group["median1"]
    obj_id_group["diffmedian12"] = obj_id_group["median1"] - obj_id_group["median2"]
    obj_id_group["diffmedian23"] = obj_id_group["median2"] - obj_id_group["median3"]
    obj_id_group["diffmedian34"] = obj_id_group["median3"] - obj_id_group["median4"]
    obj_id_group["diffmedian45"] = obj_id_group["median4"] - obj_id_group["median5"]
    obj_id_group["diffmedian04"] = obj_id_group["median0"] - obj_id_group["median4"]
    obj_id_group["diffmedian15"] = obj_id_group["median1"] - obj_id_group["median5"]

    obj_id_group["diffmean01"] = obj_id_group["mean0"] - obj_id_group["mean1"]
    obj_id_group["diffmean12"] = obj_id_group["mean1"] - obj_id_group["mean2"]
    obj_id_group["diffmean23"] = obj_id_group["mean2"] - obj_id_group["mean3"]
    obj_id_group["diffmean34"] = obj_id_group["mean3"] - obj_id_group["mean4"]
    obj_id_group["diffmean45"] = obj_id_group["mean4"] - obj_id_group["mean5"]
    obj_id_group["diffmean04"] = obj_id_group["mean0"] - obj_id_group["mean4"]
    obj_id_group["diffmean15"] = obj_id_group["mean1"] - obj_id_group["mean5"]

    obj_id_group["diffmax01"] = obj_id_group["max0"] - obj_id_group["max1"]
    obj_id_group["diffmax12"] = obj_id_group["max1"] - obj_id_group["max2"]
    obj_id_group["diffmax23"] = obj_id_group["max2"] - obj_id_group["max3"]
    obj_id_group["diffmax34"] = obj_id_group["max3"] - obj_id_group["max4"]
    obj_id_group["diffmax45"] = obj_id_group["max4"] - obj_id_group["max5"]
    obj_id_group["diffmax04"] = obj_id_group["max0"] - obj_id_group["max4"]
    obj_id_group["diffmax15"] = obj_id_group["max1"] - obj_id_group["max5"]

    obj_id_group["diffstd01"] = obj_id_group["std0"] - obj_id_group["std1"]
    obj_id_group["diffstd12"] = obj_id_group["std1"] - obj_id_group["std2"]
    obj_id_group["diffstd23"] = obj_id_group["std2"] - obj_id_group["std3"]
    obj_id_group["diffstd34"] = obj_id_group["std3"] - obj_id_group["std4"]
    obj_id_group["diffstd45"] = obj_id_group["std4"] - obj_id_group["std5"]
    obj_id_group["diffstd04"] = obj_id_group["std0"] - obj_id_group["std4"]
    obj_id_group["diffstd15"] = obj_id_group["std1"] - obj_id_group["std5"]

    obj_id_group["diffmin01"] = obj_id_group["min0"] - obj_id_group["min1"]
    obj_id_group["diffmin12"] = obj_id_group["min1"] - obj_id_group["min2"]
    obj_id_group["diffmin23"] = obj_id_group["min2"] - obj_id_group["min3"]
    obj_id_group["diffmin34"] = obj_id_group["min3"] - obj_id_group["min4"]
    obj_id_group["diffmin45"] = obj_id_group["min4"] - obj_id_group["min5"]
    obj_id_group["diffmin04"] = obj_id_group["min0"] - obj_id_group["min4"]
    obj_id_group["diffmin15"] = obj_id_group["min1"] - obj_id_group["min5"]



    obj_id_group["meanno0"] = ((obj_id_group["mean1"] + obj_id_group["mean2"] + obj_id_group["mean3"] + obj_id_group["mean4"] + obj_id_group["mean5"] )/5)
    obj_id_group["meanno1"] = ((obj_id_group["mean0"] + obj_id_group["mean2"] + obj_id_group["mean3"] + obj_id_group["mean4"] + obj_id_group["mean5"] )/5)
    obj_id_group["meanno2"] = ((obj_id_group["mean1"] + obj_id_group["mean0"] + obj_id_group["mean3"] + obj_id_group["mean4"] + obj_id_group["mean5"] )/5)
    obj_id_group["meanno3"] = ((obj_id_group["mean1"] + obj_id_group["mean2"] + obj_id_group["mean0"] + obj_id_group["mean4"] + obj_id_group["mean5"] )/5)
    obj_id_group["meanno4"] = ((obj_id_group["mean1"] + obj_id_group["mean2"] + obj_id_group["mean3"] + obj_id_group["mean0"] + obj_id_group["mean5"] )/5)
    obj_id_group["meanno5"] = ((obj_id_group["mean1"] + obj_id_group["mean2"] + obj_id_group["mean3"] + obj_id_group["mean4"] + obj_id_group["mean0"] )/5)


    obj_id_group["medianno0"] = ((obj_id_group["median1"] + obj_id_group["median2"] + obj_id_group["median3"] + obj_id_group["median4"] + obj_id_group["median5"] )/5)
    obj_id_group["medianno1"] = ((obj_id_group["median0"] + obj_id_group["median2"] + obj_id_group["median3"] + obj_id_group["median4"] + obj_id_group["median5"] )/5)
    obj_id_group["medianno2"] = ((obj_id_group["median1"] + obj_id_group["median0"] + obj_id_group["median3"] + obj_id_group["median4"] + obj_id_group["median5"] )/5)
    obj_id_group["medianno3"] = ((obj_id_group["median1"] + obj_id_group["median2"] + obj_id_group["median0"] + obj_id_group["median4"] + obj_id_group["median5"] )/5)
    obj_id_group["medianno4"] = ((obj_id_group["median1"] + obj_id_group["median2"] + obj_id_group["median3"] + obj_id_group["median0"] + obj_id_group["median5"] )/5)
    obj_id_group["medianno5"] = ((obj_id_group["median1"] + obj_id_group["median2"] + obj_id_group["median3"] + obj_id_group["median4"] + obj_id_group["median0"] )/5)

    obj_id_group["stdno0"] = ((obj_id_group["std1"] + obj_id_group["std2"] + obj_id_group["std3"] + obj_id_group["std4"] + obj_id_group["std5"] )/5)
    obj_id_group["stdno1"] = ((obj_id_group["std0"] + obj_id_group["std2"] + obj_id_group["std3"] + obj_id_group["std4"] + obj_id_group["std5"] )/5)
    obj_id_group["stdno2"] = ((obj_id_group["std1"] + obj_id_group["std0"] + obj_id_group["std3"] + obj_id_group["std4"] + obj_id_group["std5"] )/5)
    obj_id_group["stdno3"] = ((obj_id_group["std1"] + obj_id_group["std2"] + obj_id_group["std0"] + obj_id_group["std4"] + obj_id_group["std5"] )/5)
    obj_id_group["stdno4"] = ((obj_id_group["std1"] + obj_id_group["std2"] + obj_id_group["std3"] + obj_id_group["std0"] + obj_id_group["std5"] )/5)
    obj_id_group["stdno5"] = ((obj_id_group["std1"] + obj_id_group["std2"] + obj_id_group["std3"] + obj_id_group["std4"] + obj_id_group["std0"] )/5)

    obj_id_group["medianratio0"] = obj_id_group["median0"] / obj_id_group["medianno0"]
    obj_id_group["medianratio1"] = obj_id_group["median1"] / obj_id_group["medianno1"]
    obj_id_group["medianratio2"] = obj_id_group["median2"] / obj_id_group["medianno2"]
    obj_id_group["medianratio3"] = obj_id_group["median3"] / obj_id_group["medianno3"]
    obj_id_group["medianratio4"] = obj_id_group["median4"] / obj_id_group["medianno4"]
    obj_id_group["medianratio5"] = obj_id_group["median5"] / obj_id_group["medianno5"]

    obj_id_group["meanratio0"] = obj_id_group["mean0"] / obj_id_group["meanno0"]
    obj_id_group["meanratio1"] = obj_id_group["mean1"] / obj_id_group["meanno1"]
    obj_id_group["meanratio2"] = obj_id_group["mean2"] / obj_id_group["meanno2"]
    obj_id_group["meanratio3"] = obj_id_group["mean3"] / obj_id_group["meanno3"]
    obj_id_group["meanratio4"] = obj_id_group["mean4"] / obj_id_group["meanno4"]
    obj_id_group["meanratio5"] = obj_id_group["mean5"] / obj_id_group["meanno5"]

    obj_id_group["stdratio0"] = obj_id_group["std0"] / obj_id_group["stdno0"]
    obj_id_group["stdratio1"] = obj_id_group["std1"] / obj_id_group["stdno1"]
    obj_id_group["stdratio2"] = obj_id_group["std2"] / obj_id_group["stdno2"]
    obj_id_group["stdratio3"] = obj_id_group["std3"] / obj_id_group["stdno3"]
    obj_id_group["stdratio4"] = obj_id_group["std4"] / obj_id_group["stdno4"]
    obj_id_group["stdratio5"] = obj_id_group["std5"] / obj_id_group["stdno5"]

    print(np.shape(obj_id_group))

    obj_id_group.columns = list(map(''.join, obj_id_group.columns.values))

    return obj_id_group


print(bigdog.head())
