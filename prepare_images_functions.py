import cv2
import os
import glob
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imageio
import datetime
import pandas as pd
from pandas.tseries.offsets import *

def crop_images(file_path):
    ''' function to crop all original images to selected 
    areas of interest as defined by pixel ranges'''

    #numeric values for crop locations 
    pixels = [[400,800,200,600], [250,650,1250,1650],[400,800,600,1000],[550,950,1300,1700], [640,1040,250,650]]
    master = file_path                       
    os.chdir(master)
    dirs=glob.glob("*/")

    #create a subfolder for tiles 
    if not os.path.exists("Tiles"):
        os.makedirs("Tiles")

    #list of  files ending in png
    files=glob.glob("*.png")

    #crop areas of interest & save in new location 
    for i in files:
        for j in range(0,len(pixels)):
            a,b,c,d = pixels[j]
            im = cv2.imread(i)
            crop = im[a:b,c:d]
            cv2.imwrite("Tiles/{}location {}.png".format(i.replace(".png", " "),j+1),crop)
        os.chdir(master)
    return 


def augment_images(file_path, save_path, aug_range = 10):
    '''function to augment cropped images desired number of times
    Pulls images from file_path and saves augmentations to save_path'''
    
    datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1, 
                             height_shift_range=0.1,
                             shear_range=0.15, 
                             zoom_range=0.1,
                             channel_shift_range = 10, 
                             horizontal_flip=True,
                             vertical_flip = True, 
                             fill_mode = 'reflect') 
    os.chdir(file_path)
    dirs=glob.glob("*/")
    files_to_augment=glob.glob("*.png")

    for i in files_to_augment:
        image_path = '{}\{}'.format(file_path,i)
        image = np.expand_dims(imageio.imread(image_path), 0)
        datagen.fit(image)
        for x, val in zip(datagen.flow(image,                    
            save_to_dir=save_path,     
            save_prefix=i,        
            save_format='png'),range(aug_range - 1)) :  
            pass
    return 