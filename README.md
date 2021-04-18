# Satellite Image Analysis to Predict Socioeconomic Factors during the COVID-19 pandemic

The purpose of this project is to produce a feasible pipeline for predicting a custom housing market wellness index in real-time, which is especially relevant during times of disruption. This approach is scalable for predicting other socioeconomic conditions with satellite imagery.

The full report for this project can be found here: *public report pdf*.

## Getting Started

These instructions will provide step-by-step instructions on how to replicate this project on your local machine for development and testing. 

### Prerequisites

In order to deploy this code you will need the following: 

1. CSV file of external factors that can be combined with your images for prediction - see "External Factors" for a sample of what was used
2. Anaconda Environment: run [environment.yml](https://github.com/Veronica-1/Capstone/blob/main/environment.yml) in Anaconda Prompt to create a virtual environment using the code ```conda env create -f environment.yml```
3. Download all satellite images to one folder, below is a sample image for the location we used in [San Francisco](https://goo.gl/maps/V2VxX22U2857wofn7):

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/San%20Fran%20Image%20Example.png" alt="San Fran Image">
</p>

## Target Index

This project aims to predict an index representing the housing market wellness at the zip code granularity on a weekly basis. Data was downloaded from [Redfin's data center](https://www.redfin.com/news/data-center/) for the regions of interest (MA and CA) to calculate the index. The index was created from scratch due to its transparent nature and the lack of availability of indices at the appropriate level of detail for the scope of the project.

To formulate the index, monthly, zip code data was compared to monthly, county-level data. Using the relationship between these for each month, the weekly zip code data was interpolated from the weekly counted data. The various market metrics were transformed to adjust for monthly seasonality and then normalized to create the index via a linear combination of the metrics.
The index data is available at [this Tableau dashboard](https://public.tableau.com/profile/ziad.hassan#!/vizhome/CustomerInteractiveTableauLatest4_8/IndexChartDashboard) with the R code for the index's formulation available in the .html file attached in this repository.

## Process Images

The first step of implementing the code is to crop images to locations of interest (areas with high mobility). Areas that experience high levels of change - limit large structures and buildings that are stagnant. This will provide the deep learning model more opportunity to detect change in features and increase the probability that the index will be sensitive to changes in the photos. 

### Open Image Preparation.ipynb

In this notebook, change file paths where needed and test cropping to make sure the dimensions of the cuts are 400 x 400 pixels and point to desired locations in your image. 
This notebook pulls in functions from the prepare_images.py file, so make sure that it is saved in the same location as your notebook. 
Here is what that looks like on the sample image: 
<p align="center">
  <img height = 600 width = 1000 src="https://github.com/Veronica-1/Capstone/blob/main/images/High%20Mobility%20Locations.png" alt="High Mobility Locations">
</p>

If you need to change the locations of interest, you can find them in the prepare_images file. More information regarding image processing can be found [here](https://keras.io/api/preprocessing/image/) [1]. To edit or include more areas, add pixel locations in the following convention to the nested list called `pixel` - make sure that the images are all a standard shape: <br>
[x lower bound, x upper bound, y lower bound, y upper bound] 

## Image Augmentation 

Next, we need to process the crops using an image augmentation function. The purpose of this is to prevent the model from learning specific image orientations and to provide it with more data to train on. We can apply the `ImageDataGenerator` function from the Python Keras library to augment each one 10 times [2]. We defined our augmentation function to include a series of horizontal and vertical flips, edge mirroring, and frame shifting. For more information on best practices for defining your data augmentation function, see the [Keras API](https://keras.io/api/preprocessing/image/). Image augmentation should look something like this: 

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/Image%20Aug%20Horizontal.png" alt="Horizonal Image Augmentation">
</p>

After implementing this code, you should have a folder of images ready to be imported into your Python environment. The .py file called `load_data.py` contains a function that you can call with your appropriate file path. It will load and merge your images into the arrays needed as model input. Below we discuss what those structures look like. 

### Input Structure 

At this point, we can begin creating the arrays and data frames needed to train the deep learning model. Below is a diagram of the way this network will be structured. We need to generate data for the two sides of the model, the image side and the external data side.

<p align="center">
  <img height = 400 width = 300 src="https://github.com/Veronica-1/Capstone/blob/main/images/Deep%20Learning%20Model%20Structure.png" alt="Deep Learning Structure">
</p>

### Load External Data 
For the external numeric data, load your CSV file in a format that has your numeric data to be included in the prediction, as well as the index (or value to be predicted) for that week. Here is an example of our formatting: 

**IMAGE OF CSV**

In the .py file called load_data, we call the CSV files and preform merging and bucketing to prepare it for classification and align it with the photos based on date values. This function is called at the start of each model notebook, but please edit as needed directly from the .py. Outputs of the `load_data` function are an `image_array` with nested image values, and the `df_for_training` which is a df that includes only the columns you want to train your model on. One important note that is your labeling of images will depend on the convention for the satellite you're using and it will be added to during cropping and augmentation activities. Here is an example of ours: 

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/Naming%20Convention.png">
</p>

### Prep Data for Model
There is a function defined called `process_structured_data` that performs min max scaling normalization on continuous data. See [here](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/) for advice on encoding ordinal data. The function also uses quartiles to divide the index values into 4 buckets for classification in a new column. Because the purpose of this model is to inform decision-makers about the relative housing health in their community, we opted to report the index in quartiles to best demonstrate change over time without making it sensitive to minor fluctuations. Further, the buckets are more easily interpretable by stakeholders than notional index values.

After this, we loaded data into our model. Below we will cover the 3 methods we used to process images and make a prediction on the index. All model approaches encompassed the general approach used by *pyimagesearch* [4] in processing multiple inputs: images and external data. The methods below differ across their CNN structures. 


## CNN Implementation - PyImageSearch Basic Architecture
Adrian Rosebrock provides a comprehensive example with included code of mixed-date input to a deep learning model for an image regression task on [PyImageSearch](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/). He uses a database of housing photos and external factors (i.e., number of rooms, square footage, etc.) to predict the housing price. He sets up a multi-layer perceptron to process the numeric data and a CNN to process the image data, and then combines MLP and CNN output in a 2-layer feed-forward neural net to make the final prediction. We followed this basic structure for the models with our own data and made adjustments to change the model to a classification task and elimination of the initial MLP to maintain the integrity of the external data in its original intersection with the image information. [This](https://github.com/L-Lewis/Predicting-traffic-accidents-CNN) link is a project that was based on the pyimagesearch but performs classification and provides additional explanation and analysis throughout multiple model iterations. This basic CNN structure performed the worst out of all attempted architectures, with a maximum accuracy achieved of **69%** after 17 epochs of training, and early stopping occurring after 20 epochs from halted improvement in the loss function. 

<p align="center"> <b>Basic Architecture Results</b></br>
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/SimpleArchCharts.png" alt="Simple Architecture Charts">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/SimpleArchConfMat.png" alt="Simple Architecture Confusion Matrix">
</p>

## CNN Implementation - ResNet
Resnet is a state-of-the-art CNN architecture that smooths forward/backward learning through **block propogation** which diminishes the vanishing gradient issue that occurs in deep models. Vanishing gradient is a problem that occurs when the model makes diminishingly small updates to the weights of each node which essentially slows learning to a halt. This occurs when a model becomes too deep to propagate results backward through the network. Resnet models are resilient against this and also robust against over-fitting by reducing total number of trained parameters. 

We tested a basic Resnet architecture [7] and a built-in Resnet50 architecture (from Keras) for implementation with our data for bucket classification. Below are the results:

<p align="center"> <b>Basic Resnet Results</b></br>
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/ResNet1Charts.png" alt="ResNet1 Charts">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/ResNet1ConfMat.png" alt="ResNet1 Confusion Matrix">
</p>

<p align="center"> <b>Resnet50 Results</b></br>
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/ResNet50Charts.png" alt="ResNet50 Charts">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/ResNet50ConfMat.png" alt="ResNet50 Confusion Matrix">
</p>

After implementing an early stop from validation loss not improving for 10 epochs in a row, the Resnet50 trained for 38 epochs and had a maximum accuracy for predicting the test data set buckets of 49%, which occurred at the 24th epoch of training. 

## CNN Implemenetion - VGG16
Finally, we implemented a VGG16 network (architecture found [here](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)). VGG nets were developed to reduce trainable parameters by using a fixed, small kernel size to capture the same feature vectors in less time while reducing the likelihood of over-fitting. With our task, there are elements in the images that are not indicative of mobility change (buildings and roads), and a VGG is equipped to handle these since it drops "background" unnecessary image information in the last layer before prediction which increases overall accuracy [6].

<p align="center"> <b>VGG16 Results</b></br>
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/VGG16Charts_1.png" alt="VGG16 Charts">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/VGG16ConfMatrix.png" alt="VGG16 Confusion Matrix">
</p>

As for results, the VGG16 performed well despite long training times. After training this model for 30 epochs, the model was able to predict on the test set with 71% accuracy, as seen in the charts and confusion below. As evident in the accuracy chart, the validation accuracy seemed to level off at 30 epochs. Despite potential further improvement (as seen after the long stretch of accuracy hovering ~45%), the training was limited due to computation constraints. Further, this accuracy was deemed success. 

## Model Choice
Given the current dataset, the VGG16 performed optimally with an accuracy of **71%** on the test set. One notable point is that all models struggled to predict index values in bucket 4. This can likely be attributed to limited samples of images with this label in the available dataset. In future applications, this model selection might change as the dataset increases in size, more powerful computers are used, and architecture changes improve changes. Therefore, the code is structured in a way that enables easy swapping of models for the prediction itself.

## Image Prediction Notebook
As the final step, predictions on new images can be made in a separate Image Prediction Demo notebook. This notebook was used for an original demonstration of the project's functionality and can be adjusted based on stakeholder needs. The notebook consists of the following steps:

1. Load the saved model of choice from the training (VGG16 in the current iteration)
2. Retrieve the external data factors associated with the dates for the target prediction images
3. Use the model to predict the image's index classification

<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/Predict%20On%20New%20Data%20May.png" alt="Prediction Demo">
</p>

After this prediction has been run, the prediction can be compared to the historic values for the index buckets on a time series chart. This data can easily be connected to the exploration dashboard for prediction vs. historic data comparisons. In the below, chart the points in blue are real historic values for the index, and the red point is the predicted value for the week of May 7th, 2020. This visual can be recreated with new data in the image prediction notebook. 
<p align="center">
  <img src="https://github.com/Veronica-1/Capstone/blob/main/images/Wellness%20Graph.png" alt="Wellness Graph">
</p>

Finally, a .py can be run from the command prompt of the Anaconda environment created by the YML file for a "no-code" prediction straight to .csv - this is found in the file `prediction_to_csv.py`. Open up 

## Authors

* [**Jonathan Harris**](https://www.linkedin.com/in/jonathan-harris1/)
* [**Veronica Carmody**](https://www.linkedin.com/in/veronica-carmody/)

## Intellectual Property

This project is licensed under Northeastern's Intellectual Property Rights License.

## Sources Cited
1. https://keras.io/api/preprocessing/image/ 
2. https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
3. https://github.com/L-Lewis/Predicting-traffic-accidents-CNN
4. https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
5. https://github.com/L-Lewis/Predicting-traffic-accidents-CNN/blob/master/7_Model3_mixed_input_neural_network.ipynb
6. https://www.quora.com/What-is-the-VGG-neural-network
7. https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/

## Acknowledgments

* We'd like to acknowledge the author of *pyimagesearch* for providing open source code that informed the initial understanding of a mixed-data approach and other authors of the cited references.
* We'd also like to acknowledge our advisor, Sagar Kamarthi PhD, and our technical design reviewer, Sri Radhakrishnan PhD.
* Finally, we'd like to thank our Northeastern capstone advisors, Professors McManus & Jager-Helton of the IE department!
