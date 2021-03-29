# IE Capstone 2021: Satellite Image Analysis to Predict Housing Wellness

The purpose of this project ... 

**Results**: results overview 

## Getting Started

These instructions will provide step-by-step instructions on how to replicate this project on your local machine for development and testing. 

### Prerequisites

In order to deploy this code you will need the following: 

1. CSV file of external factors that can be combined with your images for prediction - see "External Factors" for a sample of what we used
2. Anaconda Environment: run [environment.yml](https://github.com/Veronica-1/Capstone/blob/main/environment.yml) in Anaconda Prompt to create a virtual environment using the code ```conda env create -f environment.yml```
3. Download all satellite images to one folder, below is a sample image for the location we used in [San Francisco](https://goo.gl/maps/V2VxX22U2857wofn7):

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/San%20Fran%20Image%20Example.png" alt="San Fran Image">
</p>

## Process Images

The first step of implementing the code is to process satellite images using an image augmentation function. The purpose of this is to prevent the model from learning specific image orientations and to provide more data to train on. 

### Open the Image Processing Notebook .ipynb 

In this notebook, change filepaths where needed and test cropping to make sure the dimensions of the cuts are 400 x 400 pixels and point to locations of high mobility in your image. 
Here is what it looks like on our sample image: 
<p align="center">
  <img height = 500 width = 1000 src="https://github.com/Veronica-1/Capstone/blob/main/High%20Mobility%20Locations.png" alt="High Mobility Locations">
</p>

Function to crop each image at the same points of interest. <br>
To include more points, add pixel locations in the following convention to the nested list called `pixel`: [x lower bound, x upper bound, y lower bound, y upper bound] 
```
# define pixel locations for each crop on the image 
pixels = [[500,900,250,650], [300,700,1300,1700],[300,700,600,1000],[575,975,1300,1700], [640,1040,250,650]]
```

### Perform Image Augmention 

This portion of the code applies the `ImageDataGenerator` function from the Python Keras library on the cropped images to augment each one 10 times. The function currently is as follows and it includes both horizonal and vertical flips as well as reflection of the edges during shifts to maximize change captured between photos. 

```
# datagen is the function used to augment the images [1] 
# choose fill_mode from reflect, nearest, constant, or wrap 

datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1, 
                             height_shift_range=0.1,
                             shear_range=0.15, 
                             zoom_range=0.1,
                             channel_shift_range = 10, 
                             horizontal_flip=True,
                             vertical_flip = True, 
                             fill_mode = 'reflect')
```
After defining the augmentation function, perform the augmentation on each of the images [2]. You should now have a folder of images ready to be imported into your Python environment. 

## Create Arrays for CNN 

At this point, we can begin creating the arrays and dataframes needed to train the deep learning model. Below is a diagram of the way this network will be structured. We need to generate data for the two sides of the model, the image side and the external data side.

<p align="center">
  <img height = 500 width = 300 src="https://github.com/Veronica-1/Capstone/blob/main/Deep%20Learning%20Model%20Structure.png" alt="Deep Learning Structure">
</p>

### Preprocess Data
For the external numeric data, load your CSV file - be sure to drop duplicate weeks and to round each week to the prior **Sunday**. Then divide the index values into buckets using XXX formula so that the CNN can be tested with both regression and classification. `image_array` and `image_labels` are an array and a list respectively read from the folder where the augemented images reside on your machine. Extract the date from the label of the satellite photo - naming convention is as follows - round date to prior **Sunday** and inner join with external data on `match_date`:

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/Naming%20Convention.png" alt="Naming Convention">
</p>

## CNN Implementation - ResNet (Link to that File)
In model one, the structure of the CNN is a classification based on code from a RestNet architecture
Here are the results for model accuracy & loss: 

** pic ** 

## CNN Implemenetion - VGG16
In model two, we tried a VGG16 architecture and here are the results: 

## CNN Implementation - PyImageSearch
In model three, we tried an architecture from pyImage search



### Versioning
In the files of this repo, you can find 3 notebooks for each of the models, a notebook showing how to implement a trained model on new data, and finally background code briefly explaining how the index we used was developed. 


## Authors

* **Jonathan Harris**   - [LinkedIn](https://www.linkedin.com/in/jonathan-harris1/)
* **Veronica Carmody**  - [LinkedIn](https://www.linkedin.com/in/veronica-carmody/)
* Rest of Team 

## Intellectual Property

This project is licensed under Northeastern Intellectual Property Rights License 

## Sources Cited
1. https://keras.io/api/preprocessing/image/ 
2. https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
3. https://github.com/L-Lewis/Predicting-traffic-accidents-CNN
4. https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
