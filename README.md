# IE Capstone 2021: Satellite Image Analysis to Predict Housing Wellness

The purpose of this project ... 

**Results**: results overview 

## Getting Started

These instructions will provide step-by-step instructions on how to replicate this project on your local machine for development and testing purposes. 

### Prerequisites

In order to deploy this code you will need the following: 

1. Downloaded CSV file with external factors that can be combined with image data to predict index
2. Downloaded folder that contains each satellite image used in prediction with consistent naming convention
3. Anaconda Environment --> run environment.yml file in Anaconda Prompt in order to create the virtual environment to run the code ```conda env create -f environment.yml```

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Process Images

The first step of implementing the code is to process satellite images using an image augmentation function. 
The purpose of this is to: 
1. Prevent the model from learning specific image orientations
2. Provide more data to train on 

### Open the Image Processing Notebook .ipynb 

In this notebook, change filepaths where needed and test cropping to make sure the dimensions of the cuts are 400 x 400 pixels. 
**Example:**
```
# define pixel locations for each crop on the image 
pixels = [[0,100,0,100], [0,200,0,200],[0,300,0,300],[0,400,0,400]]

master = r"...\Original_Images"                       
os.chdir(master)
dirs=glob.glob("*/")

#create a subfolder for tiles 
if not os.path.exists("Tiles"):
    os.makedirs("Tiles")

#list of  files 
files=glob.glob("*.png")

#loop through files, crop areas of interest & save in tiles 
for i in files:
    for j in range(0,len(pixels)):
        a,b,c,d = pixels[j]
        im = cv2.imread(i)
        crop = im[a:b,c:d]
        cv2.imwrite("Tiles/{}location {}.png".format(i.replace(".png", " "),j+1),crop)
    os.chdir(master) 
```

### Perform Image Augmention 

This portion of the code applies the ImageDataGenerator function from the Python Keras library on the images to augment each one 10 times. 
The function currently is as follows and it includes both horizonal and vertical flips as well as reflection of the edges during shifts to maximize change captured between photos. 

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
