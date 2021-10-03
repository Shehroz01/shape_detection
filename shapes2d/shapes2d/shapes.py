#!/usr/bin/env python3

# numpy and matplotlib are the required libraries for this task
# matplotlib for visualization and numpy for mathematical computation. (Does pycharm support plots with plt.imshow()?)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


##########################################################################


# Create a dataset of images on 32*32 grid with 2D geometric shapes placed on it.
# This task would be for a single shape, a rectangle. Also we need to define the bounding boxes for detection.

# Size of the dataset of images and bounding boxes
dataset_size = 10000

# Grid size required is 32*32
grid_size = 32

# Number of shapes in the dataset
num_shapes = 1

# Maximum size of the object in the dataset
object_max = 10

# Minimum size of the object in the dataset
object_min = 4


# Create the dataset with 2D rectangles on the grid. The rectangle is assigned 1 while background is 0.
dataset = np.zeros((dataset_size, grid_size, grid_size))

# Create bounding boxes equal to the dataset of images
bound_box = np.zeros((dataset_size, num_shapes, 4))


# Iterating over the length of dataset and generating random rectangles and bounding boxes
for k in range(dataset_size):
    for l in range(num_shapes):
        w, h = np.random.randint(object_min, object_max, size=2)
        x = np.random.randint(0, grid_size - w)
        y = np.random.randint(0, grid_size - h)

        # TODO: Verify rectangle is correctly set to 1
        dataset[k, x:x + w, y:y + h] = 1.
        bound_box[k, l] = [x, y, w, h]

# print(dataset.shape)
# print(bound_box.shape)


###############################################################################

# Visualizing a generated image of the dataset

image = 10
plt.imshow(dataset[image].T, cmap='PuBu', interpolation='none', origin='lower', extent=[0, grid_size, 0, grid_size])
for box in bound_box[image]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((box[0], box[1]), box[2], box[3], ec='g', fc='b'))

plt.show(block=True)
plt.interactive(False)


################################################################################

# Preprocessing the dataset before the training

# Reshape the image data
images = (dataset.reshape(dataset_size, -1) - np.mean(dataset)) / np.std(dataset)
# print(images.shape)
# print(np.mean(images))
# print(np.std(images))


# Normalize the xy coordinates, width, height, by grid_size (values between 0 and 1).
boxes = bound_box.reshape(dataset_size, -1) / grid_size
# print(boxes.shape)
# print(np.mean(boxes))
# print(np.std(boxes))

################################################################################

# Dividing the image data for training and testing.

# 75% data for training and 25% for testing
ratio = int(0.75 * dataset_size)
train_images = images[:ratio]
test_images = images[ratio:]
train_boxes = boxes[:ratio]
test_boxes = boxes[ratio:]
test_dataset = dataset[ratio:]
test_bound_box = bound_box[ratio:]


################################################################################

# Defining the model.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

model = Sequential([
        Dense(32, input_dim=images.shape[-1]), # TODO: Optimize
        Activation('relu'),
        Dropout(0.1), # TODO: Optimize
        Dense(boxes.shape[-1])
    ])
model.compile('adam', 'mse') # adam works very well generally.


# Training the model
model.fit(train_images, train_boxes, nb_epoch=100, validation_data=(test_images, test_boxes), verbose=2)


###############################################################################

# Generating/estimating bounding boxes on test dataset
gen_boxes = model.predict(test_images)
gen_bound_box = gen_boxes * grid_size
gen_bound_box = gen_bound_box.reshape(len(gen_bound_box), num_shapes, -1)
# print(gen_bound_box.shape)

###############################################################################