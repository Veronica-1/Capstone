# Satellite Image Analysis to Predict Real Estate Housing Wellness

The purpose of this project is to produce a feasible pipeline for predicting a custom housing market wellness index in real-time, which is especially relevant during times of disruption. This approach is scalable for predicting other socioeconomic conditions with satellite imagery.

The full report for this project can be found here: PUBLIC WORD DOC, PDF, OR PDF UPLOADED TO THIS GITHUB PAGE.

**Results**: results overview 

## Getting Started

These instructions will provide step-by-step instructions on how to replicate this project on your local machine for development and testing. 

### Prerequisites

In order to deploy this code you will need the following: 

1. CSV file of external factors that can be combined with your images for prediction - see "External Factors" for a sample of what was used
2. Anaconda Environment: run [environment.yml](https://github.com/Veronica-1/Capstone/blob/main/environment.yml) in Anaconda Prompt to create a virtual environment using the code ```conda env create -f environment.yml```
3. Download all satellite images to one folder, below is a sample image for the location we used in [San Francisco](https://goo.gl/maps/V2VxX22U2857wofn7):

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/San%20Fran%20Image%20Example.png" alt="San Fran Image">
</p>

## Target Index

This project aims to predict an index representing the housing market wellness at the zip code granularity on a weekly basis. Data was downloaded from [Redfin's data center](https://www.redfin.com/news/data-center/) for the regions of interest (MA and CA) to calculate the index. The index was created from scratch due to its transparent nature and the lack of availability of indicies at the appropriate level of detail for the scope of the project.

To formulate the index, monthly, zip code data was compared to monthly, county-level data. Using the relationship between these for each month, the weekly zip code data was interpolated from the weekly counted data. The various market metrics were normalized to create the index via a linear combination of the metrics.
The index data is available at <b><font color="red">THIS TABLEAU DASHBOARD</font></b> with the R code for the index's formulation available in the <b><font color="red">respective file included</font></b>.

## Process Images

The first step of implementing the code is to crop images to locations of interest (areas with high mobility). Areas that experience high levels of change - limit large structures and buildings that are stagnant. This will provide the deep learning model more opportunity to detect change in features and increase the probability that the index will be sensitive to changes in the photos. 

### Open the Image Processing Notebook .ipynb

In this notebook, change filepaths where needed and test cropping to make sure the dimensions of the cuts are 400 x 400 pixels and point to desired locations in your image. 
Here is what that looks like on the sample image: 
<p align="center">
  <img height = 600 width = 1000 src="https://github.com/Veronica-1/Capstone/blob/main/High%20Mobility%20Locations.png" alt="High Mobility Locations">
</p>

To include more areas of interest, add pixel locations in the following convention to the nested list called `pixel` in the .ipynb called Process Images: <br>
[x lower bound, x upper bound, y lower bound, y upper bound] 

## Open the Image Augmentation Notebook .ipynb 

Next, we need to process the crops using an image augmentation function. The purpose of this is to prevent the model from learning specific image orientations and to provide it with more data to train on. We can apply the `ImageDataGenerator` function from the Python Keras library to augment each one 10 times [2]. We defined our augmentation function to include a series of horizontal and vertical flips, edge mirroring, and frame shifting. For more information on best practices for defining your data augmention function, see the [Keras API](https://keras.io/api/preprocessing/image/). Image augmentation should look something like this: 

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/Image%20Aug%20Horizontal.png" alt="Horizonal Image Augmentation">
</p>

After implementing this code, you should have a folder of images ready to be imported into your Python environment. The .py file called `load_data.py` contains a function that you can call with your appropriate file path. It will load and merge your images into the arrays needed as model input. Below we discuss what those structures look like. 

### Input Structure 

At this point, we can begin creating the arrays and dataframes needed to train the deep learning model. Below is a diagram of the way this network will be structured. We need to generate data for the two sides of the model, the image side and the external data side.

<p align="center">
  <img height = 500 width = 300 src="https://github.com/Veronica-1/Capstone/blob/main/Deep%20Learning%20Model%20Structure.png" alt="Deep Learning Structure">
</p>

### Load External Data 
For the external numeric data, load your CSV file - be sure to drop duplicate weeks and to round each week to the prior **Sunday**. If you plan on testing a classification model, as done in this model, divide the index values into buckets. You will need to normalize your external data. There is a function defined called `process_structured_data` that performs min max scaling on continuous data. See [here](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/) for advice on encoding ordinal data. 

### Load Image Data 
`image_array` and `image_labels` are an array and a list, respectively, read from the folder where the augemented images reside on your machine. Extract the date from the label of the satellite photo - naming convention is as follows - and round the date to prior **Sunday** to inner join with external data on `match_date`.

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/Naming%20Convention.png">
</p>

After this, you should be ready to load your data in a model. Below we will cover the 4 methods we used to process images and make a prediction on the index. 1 is regression and 3 are classification and 1 is regression. All model approaches encompassed the general approach used by *pyimagesearch* [4] in processing multiple inputs: images and other data. The methods below differ across their CNN structures. 

## CNN Implemention - Regression 
Discussion of results 

## CNN Implementation - ResNet
In Model 1, the structure of the CNN is a classification based on code from a RestNet architecture.
Here are the results for model accuracy & loss: 

## CNN Implemenetion - VGG16
In Model 2, we tried a VGG16 architecture. The results were as follows:

## CNN Implementation - PyImageSearch Architecture
In Model 3, we tried an architecture from pyImage search. This was the best performing model. The results were as follows:


### Versioning


## Authors

* [**Jonathan Harris**](https://www.linkedin.com/in/jonathan-harris1/)
* [**Veronica Carmody**](https://www.linkedin.com/in/veronica-carmody/)
* Rest of Team 

## Intellectual Property

This project is licensed under Northeastern's Intellectual Property Rights License.

## Sources Cited
1. https://keras.io/api/preprocessing/image/ 
2. https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
3. https://github.com/L-Lewis/Predicting-traffic-accidents-CNN
4. https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
