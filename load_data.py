import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import imageio
import datetime
from pandas.tseries.offsets import *
from matplotlib.pyplot import imshow, show
import matplotlib as plt
import pandas as pd

def load_data_MA(aug_location, external_path):
    '''load image labels from file path, 
    get date from the label, convert to Sunday
    sort on (label, match_date)
    load external data, drop duplicates, convert to Sunday
    bucket external data on percentiles
    inner merge external data and labels,
    sort on (label, match_date)
    read in images as an array in the order & amount of inner join result
    return 1) image array 2) merge_with_labels'''

    #go to location of augmented images
    os.chdir(aug_location)

    #create list of all photos in the file 
    dirs=glob.glob("*/")
    files_to_array=glob.glob("*.png")

    #image labels 
    image_labels = []

    #append opened images based on file name & collect labels in a list (maintains order)
    for i in files_to_array:
        image_labels.append(str(i))

    #convert to array
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
    
    #Sort on Label & match_date 
    file_df = file_df.sort_values(["Label", "match_date"])

    #Import external data and create a column that rounds date to prior Sunday & drop duplicates
    external_data = pd.read_csv(external_path)
    external_data['Date'] = pd.to_datetime(external_data['Date'])
    external_data["match_date"] =  external_data['Date'] + Week(weekday=6) - Week()
    external_data = external_data.drop_duplicates(subset="match_date")

    #Create buckets based on percentiles 
    external_data["buckets"] = pd.qcut(external_data["MA_3Wk"], 4, labels = ["0", "1", "2", "3"])
    
    #Merge data 
    merge_with_labels = pd.merge(file_df[["Label","match_date"]],external_data.iloc[:,2:], on = ["match_date"], how = "inner")

    #Sort on Label & match_date
    merge_with_labels = merge_with_labels.sort_values(["Label", "match_date"])

    #Get df_for_training
    df_for_training = merge_with_labels [["Mortgage Rate" , "Percent Delinquent Mortgages", "Unemployment"]].copy()
    
    #Read in array in the same order as merge_with_labels
    desired_files = list(merge_with_labels["Label"])

    #nested image array
    image_array = []

    #loop through images and add to array
    for i in desired_files:
        image_array.append(np.array(Image.open('{}\{}'.format(aug_location,i))))

    image_array = np.asarray(image_array)
    
    return image_array, merge_with_labels, df_for_training


def prediction_fxn(array_input):
    'Predicted Bucket on 1 - 4 scale'
    result = array_input.argmax() + 1
    if result == 1:
        percentile = "0 - 25th percentile"
    elif result == 2:
        percentile = "25th - 50th percentile"
    elif result == 3: 
        percentile = "50th - 75th percentile"
    else:
        percentile = "75th - 100th percentile"
    return result, percentile

def actual_percentile(input):
    'Actual Bucket on 1 - 4 scale'
    input = int(input)
    if input == 0:
        percentile = "0 - 25th percentile"
    elif input == 1:
        percentile = "25th - 50th percentile"
    elif input == 2: 
        percentile = "50th - 75th percentile"
    else:
        percentile = "75th - 100th percentile"
    return input, percentile

def show_predictions(image_array_demo,merge_with_labels_demo,prediction):
    for i in range(0,len(image_array_demo)):
        plt.pyplot.imshow(image_array_demo[i])
        show()
        date_time = merge_with_labels_demo["Date"][i]
        date = date_time.to_pydatetime()
        print("Date:",date.date())
        print("Predicted Value:", prediction_fxn(prediction[i])[0])
        print("Predicted Percentile:", prediction_fxn(prediction[i])[1])
        print("Actual Value:", int(merge_with_labels_demo["buckets"][i]) + 1)
        print("Actual Percentile:", actual_percentile(merge_with_labels_demo["buckets"][i])[1])
    return

def array_to_df(prediction_array):
    df_list = []
    for i in range(0,len(prediction)):
        val = prediction_fxn(prediction_array[i])[0]
        df_list.append(val)
    df_output = pd.DataFrame({'predicted_bucket':df_list})
    return df_output
