
# Detection of 2D Geometric Shapes

The main goal of this project is to detect 2D geometric shapes using tensorflow and python.

The strategy is to break down the problem statement into small and precise chunks, so that the
requirements are well defined. There are 3 major steps involved:

* Setting up the environment for the programming project
* Creating a dataset of geometric shapes
* Training a machine learning model with this dataset

The environment I'm using is with Poetry and GitHub is the host for this project. The dataset can 
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
* Set up a virtual environment with compatible packages. Avoid installing packages through pip
* Manage dependencies and PYTHONPATH errors (solved after a headache)


## Step 4: Creating a dataset with 2D shapes, partial occlusions

Requirements: 
* Dataset having a 32*32 grid with 2D geometric shapes
* Partial occlusions on the top of geometric shapes

For the first step, a model that detects a single geometric shape would be a good start. For this purpose, I have
selected a dataset of rectangles. The idea is to generate random rectangles on the 32*32 grid and create a dataset of 10000 images.
Also generate bounding boxes in the same quantity of image dataset.


![geometric](https://user-images.githubusercontent.com/52299886/135727480-8be49e30-2068-47d6-8e3a-c756f216b8c1.PNG)



The dataset needs to be reshaped and normalized before it could be split into train and test dataset. (working on it) 
 

## Step 5: Creating a model for predicting geometric shapes

(Need to brainstorm and identify the requirements)


A problem well defined is a problem half solved!
