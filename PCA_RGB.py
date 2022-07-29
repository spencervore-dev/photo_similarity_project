from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn import manifold
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import imageio as iio
import math
import pandas as pd
import cv2
import scipy.io
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

#############################################################
# read a series of images in a folder and save them in a list
#############################################################

## Configuration
# filepath = "/home/spencervore/OneDrive/isye6740_project_data/pexel/green_medium/*.jpeg"
#filepath = "/home/spencervore/OneDrive/isye6740_project_data/pexel/test_mojtaba/*.jpeg"
#filepath = "/home/spencervore/OneDrive/isye6740_project_data/pixabay/green/*.jpg"

filepath = r'/home/spencervore/OneDrive/isye6740_project_data/pexel/green_medium'
make_plots = True

#################################################################
# Find minimum dimension of the images (x or y dim) in the folder
#################################################################
print("Finding min dimensions")
dimensions = []
for img in os.listdir(filepath):
    dimensions.append(cv2.imread(os.path.join(filepath, img)).shape[0])
    dimensions.append(cv2.imread(os.path.join(filepath, img)).shape[1])
min_dimension = min(dimensions)

#################################################################
# zero padding to make the images square
#################################################################
print("Zero padding and standardizing image sizes")
images = []
for filename in os.listdir(filepath):
    rgb = cv2.imread(os.path.join(filepath, filename))
    diff = max(rgb.shape[0:2]) - min(rgb.shape[0:2])  # find the number of zero pad
    zpad = int(diff / 2) + 1
    x, y = rgb.shape[1], rgb.shape[0]

    # decide which dimension to zero pad, and adjust if the dimension difference is even or odd
    if x < y:
        if (diff % 2) == 0:
            rgb = np.pad(rgb, pad_width=[(0, 0), (zpad, zpad), (0, 0)], mode="constant")
        else:
            rgb = np.pad(rgb, pad_width=[(1, 0), (zpad, zpad), (0, 0)], mode="constant")
    elif x > y:
        if (diff % 2) == 0:
            rgb = np.pad(rgb, pad_width=[(zpad, zpad), (0, 0), (0, 0)], mode="constant")
        else:
            rgb = np.pad(rgb, pad_width=[(zpad, zpad), (1, 0), (0, 0)], mode="constant")
    else:
        # Case if image is perfect square already... still need to convert to numpy
        # so everything comes out in same format
        rgb = np.array(rgb)

    ######### cut the images to the minimum dimension of the image in the folder
    ######### to make them all the same size.
    rgb_shape = rgb.shape[0]
    lower_cut = (rgb_shape - min_dimension) / 2
    upper_cut = rgb_shape - (rgb_shape - min_dimension) / 2

    # I think the above was making this come out as two different sizes. For this to work, everything must be exactly
    # the same size no matter how it's processed.
    rgb = rgb[int(lower_cut):int(upper_cut), int(lower_cut):int(upper_cut), ]

    images.append(rgb.flatten())

images = np.array(images)
images_count = len(images)

#####################################################
######## PCA with 2 PCs
#####################################################
print("Run PCA")
pca = PCA(n_components=2)
pca_imgs = pca.fit_transform(images)
pic_array = np.array(pca_imgs)

x = pic_array[0]
y = pic_array[1]

####################################
########### Plotting ###############
####################################
print("Plotting results")
if make_plots:
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    ax = fig.add_subplot(111)
    plt.title('2D Plot Using PCA of RGB Animal Images')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # plot PCA results
    num_images = images_count
    image_dimensions = (min_dimension, min_dimension, 3)
    x_size = (max(x) - min(x)) * 0.25
    y_size = (max(y) - min(y)) * 0.1

    pic = 0
    for p in pic_array:
        x0 = p[0] - (x_size / 2.)
        x1 = p[0] + (x_size / 2.)
        y0 = p[1] - (y_size / 2.)
        y1 = p[1] + (y_size / 2.)
        img = images[pic].reshape(image_dimensions)
        ax.imshow(img, aspect='auto', interpolation='nearest',
                  zorder=100000, extent=(x0, x1, y0, y1))

        pic += 1
    # Show 2D components plot
    ax.scatter(pic_array[0], pic_array[1], marker='.', alpha=0.7)

    plt.show()
