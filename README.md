
# Detection of 2D Geometric Shapes

The main goal of this project is to detect 2D geometric shapes using tensorflow and python.

The strategy is to break down the problem statement into small and precise chunks, so that the
requirements are well defined. There are 3 major steps involved:

* Setting up the environment for the programming project
* Creating a dataset of geometric shapes
* Training a machine learning model with this dataset

The environment I'm using is Poetry and GitHub is the host for this project. The dataset can 
consist of multiple geometric shapes but for simplicity I would divide this tasks into further small 
steps:

* Machine learning model to detect a single geometric shape
* Machine learning model with multiple shapes
* Machine learning model with multiple shapes and partial occlusions in the dataset 

## Step 1: Creating a github project (Completed)

Created and shared with the team.



## Step 2: Creating a Poetry Project (Completed)

* Why Poetry? 
It is a tool for dependency management and packaging in Python, therefore, we can create
reproducible build, so other can recreate it on their own. 

* What do I need to know about Poetry?
How it works, how to install it, and how package dependencies are managed. 

* How does it work with Windows?
It can be installed with Powershell, and it worked. Created a new project.

* Think about the required dependencies in the next steps (solved)



## Step 3: Setting up a tensorflow machine learning project using keras (Completed)

* Which libraries do I need? (matplotlib, numpy, tensorflow)
* Set up a virtual environment with compatible packages. Avoid installing packages through pip, only install when necessary
* Manage dependencies and PYTHONPATH errors (solved)



## Step 4: Creating a dataset with 2D shapes (Completed)

Requirements: 
* Dataset having a 32*32 grid with 2D geometric shapes
* Partial occlusions on the top of geometric shapes (next step)

For the first step, a model that detects a single geometric shape would be a good start. For this purpose, I have
selected a dataset of rectangles. The idea is to generate random rectangles on the 32*32 grid and create a dataset of 10000 images.
Also generate bounding boxes for image dataset. An example of such an image can be seen below.


![geometric](https://user-images.githubusercontent.com/52299886/135727480-8be49e30-2068-47d6-8e3a-c756f216b8c1.PNG)



In the above figure, the rectangle is defined as a 1 whereas the background is zero. There is a bounding box in green color around the blue rectangle. The dataset size of such images and bounding boxes generated is 10000. The training dataset would be passed on to the model to learn and then predict the position of the real rectangle (generated) with the test dataset. The dataset needs to be reshaped (image dataset) and normalized (generated bounding boxes) before it could be split into train and test dataset.  
 


## Step 5: Creating a model for predicting geometric shapes (Completed)

For this task, I'm selecting a fairly simple model. The dataset isn't complex as compared to images of real objects such
as cars or cats. For the dense layer, I will start with 32 and then play around to see if the results worsens or gets better.
Similarly for the dropout value, I will select 0.1 and then gradually increase. Although, optimization might not matter for this
task that much but in general, you have to develop some insights building a model especially when you are working with large
real-image dataset. 

For the activation function, I have selected the rectified linear activation function or ReLU. It is a piecewise linear function 
that pass the input if it is positive, or zero if it's negative. It has become a standard function for many types of neural
networks because it is easy to train and gives good results. For the optimizer, I have selected 'adam' because it works pretty well
for a variety of applications. I will try to use other optimizers for the next steps but I think it's not required.

And finally, I'm using 'mse' or the Mean Square Error. It is the mean or average of the square of the difference between real 
and predicted values. Similarly for training the model, I trained for 100 epochs. Both the training loss and the validation 
loss decreases with the number of epochs increasing. On my system, it takes around 1-2 minutes for 10,000 image and bounding boxes dataset.



![training](https://user-images.githubusercontent.com/52299886/135758649-213587a8-0806-4a7e-b33f-68cf4da2a1a2.PNG)


* Predict bounding boxes for the test dataset

After the model is trained, the test images can be passed to predict the shape in the image. The generated bounding boxes need to be 
scaled according to the grid size (32) and then reshaped.

## Next Steps for this project:

* Design a method/function for IOU calculation to calculate the overlapping between the real shape (test image) and the predicted bounding box.
* Introduce multiple shapes in the dataset and add partial occlusions   

A problem well defined is a problem half solved!
