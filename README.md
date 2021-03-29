# IE Capstone 2021: Satellite Image Analysis to Predict Housing Wellness

The purpose of this project ... 

**Results**: results overview 

## Getting Started

These instructions will provide step-by-step instructions on how to replicate this project on your local machine for development and testing. 

### Prerequisites

In order to deploy this code you will need the following: 

1. Downloaded CSV file with external factors that can be combined with image data to predict index
2. Downloaded folder that contains each satellite image used in prediction with consistent naming convention
3. Anaconda Environment --> run environment.yml file in Anaconda Prompt in order to create the virtual environment to run the code ```conda env create -f environment.yml```

Here is a sample image for the location we used located [here](https://goo.gl/maps/V2VxX22U2857wofn7) in San Francisco: 

![Satellite Image Example](https://github.com/[Veronica-1]/[Capstone]/blob/[main]/San Fran Image Example.png)

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

# create a subfolder for tiles 
if not os.path.exists("Tiles"):
    os.makedirs("Tiles")

# list of  files 
files=glob.glob("*.png")

# loop through files, crop areas of interest & save in tiles 
for i in files:
    for j in range(0,len(pixels)):
        a,b,c,d = pixels[j]
        im = cv2.imread(i)
        crop = im[a:b,c:d]
        cv2.imwrite("Tiles/{}location {}.png".format(i.replace(".png", " "),j+1),crop)
    os.chdir(master) 
```

### Perform Image Augmention 

This portion of the code applies the ImageDataGenerator function from the Python Keras library on the cropped images to augment each one 10 times. 
The function currently is as follows and it includes both horizonal and vertical flips as well as reflection of the edges during shifts to maximize change captured between photos. 

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
After defining the augmentation function, perform the augmentation on each of the images with the following nested loop: 

```
master = r"...\Location_of_Cropped_Images"
save_here = r"...\Location_to_Save_Images"
os.chdir(master)
dirs=glob.glob("*/")
files_to_augment=glob.glob("*.png")

# function to augment each image 10 times (range 9) and keep naming convention constant 
for i in files_to_augment:
    image_path = '{}\{}'.format(master,i)
    image = np.expand_dims(imageio.imread(image_path), 0)
    datagen.fit(image)
    for x, val in zip(datagen.flow(image,                    
        save_to_dir=save_here,     
        save_prefix=i,        
        save_format='png'),range(9)) :  
        pass
 ```
 After this, you should have a folder of images ready to be imported into your Python environment. 

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

* **Veronica Carmody**  - [LinkedIn](https://www.linkedin.com/in/veronica-carmody/)
* **Jonathan Harris**   - [LinkedIn](https://www.linkedin.com/in/jonathan-harris1/)
* Rest of Team 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Sources Cited
[1] https://keras.io/api/preprocessing/image/
[2] 

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
