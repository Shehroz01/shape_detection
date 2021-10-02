#!/usr/bin/env python3

# Numpy and Matplotlib are the required libraries for this task
# Matplotlib for visualization and Numpy for mathematical computation. (Does pycharm support plots with plt.imshow()?)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


##########################################################################################


# Create a dataset of images with 32*32 grid with 2D geometric shapes placed on it.
# This task would be for a single shape, a rectangle. Also we need to define the bounding boxes for detection.

# Size of the dataset of images
dataset_size = 5000

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


# Iterating over the range of dataset and generating random rectangles and bounding boxes
for k in range(dataset_size):
    for l in range(num_shapes):
        w, h = np.random.randint(object_min, object_max, size=2)
        x = np.random.randint(0, grid_size - w)
        y = np.random.randint(0, grid_size - h)

        # TODO: Verify rectangle is correctly set to 1
        dataset[k, x:x + w, y:y + h] = 1.
        bound_box[k, l] = [x, y, w, h]

print(dataset.shape)
print(bound_box.shape)


# Plotting dataset images
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
print(images.shape)
print(np.mean(images))
print(np.std(images))


# Normalize the xy coordinates, width, height, by grid_size (values between 0 and 1).
boxes = bound_box.reshape(dataset_size, -1) / grid_size
print(boxes.shape)
print(np.mean(boxes))
print(np.std(boxes))

################################################################################




