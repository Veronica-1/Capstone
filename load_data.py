import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import imageio
import datetime
from pandas.tseries.offsets import *


def load_data(aug_location, external_path):
    '''function to load nested array of images & 
    load image labels from augmented tiles, extract date from labels 
    load external data, bucket Wellness Index by percentiles and
    merge dataframes on date, ouput is nested array and df with columns for training'''
    
    #go to location of augmented images
    os.chdir(aug_location)

    #create list of all photos in the file 
    dirs=glob.glob("*/")
    files_to_array=glob.glob("*.png")

    #nested image array
    image_array = []

    #image labels 
    image_labels = []

    #append opened images based on file name & collect labels in a list (maintains order)
    for i in files_to_array:
        image_array.append(np.array(Image.open('{}\{}'.format(aug_location,i))))
        image_labels.append(str(i))

    #convert to array
    image_array = np.asarray(image_array) 
    image_labels= np.asarray(image_labels)
    
    #Extract date from image label
    file_df = pd.DataFrame(image_labels, columns = ["Label"])

    file_df["year"] = file_df.Label.str[16:20]
    file_df["month"] = file_df.Label.str[20:22]
    file_df["day"] = file_df.Label.str[22:24]

    file_df = file_df.astype({"year": int, "month": int, "day":int})

    file_df["date"] = file_df.apply(lambda x: datetime.date(x['year'], x['month'], x['day']), axis=1)
    file_df['date'] = pd.to_datetime(file_df['date'])

    #Convert Date to a Sunday Date 
    file_df["match_date"] =  file_df['date'] + Week(weekday=6) - Week()

    #Import external data and create a column that rounds date to prior Sunday & drop duplicates
    external_data = pd.read_csv(external_path)

    external_data['Date'] = pd.to_datetime(external_data['Date'])

    external_data["match_date"] =  external_data['Date'] + Week(weekday=6) - Week()

    external_data = external_data.drop_duplicates(subset="match_date")

    #Create buckets based on percentiles 
    external_data["buckets"] = pd.qcut(external_data["Index Wellness"], 4, labels = ["0", "1", "2", "3"])
    
    #Merge data 
    merge_with_labels = pd.merge(file_df[["Label","match_date"]],external_data.iloc[:,2:], on = ["match_date"], how = "inner")

    #Grab columns for training 
    df_for_training = merge_with_labels[["Mortgage Rate" , "Percent Delinquent Mortgages", "Unemployment", "buckets"]].copy()
    
    return image_array, df_for_training