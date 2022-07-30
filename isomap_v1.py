#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 03:46:56 2022

@author: mojtabataghipourkaffash
"""

from sklearn import manifold
import imageio as iio
import pandas as pd
import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


#############################################################
# read a series of images in a folder and save them in a list
#############################################################

## Configuration
# filepath = "/home/spencervore/OneDrive/isye6740_project_data/pexel/green_medium/*.jpeg"
# filepath = "/home/spencervore/OneDrive/isye6740_project_data/pexel/test_mojtaba/*.jpeg"
filepath = "/home/spencervore/OneDrive/isye6740_project_data/pixabay/transparent/*.png"

make_plots = True



#################################################################
# Find minimum dimension of the images (x or y dim) in the folder
#################################################################
print("Finding min dimensions")
dimensions = []
n_layers = 0
for filename in glob.glob(filepath):
    filedata = iio.v3.imread(filename)
    dimensions.append(filedata.shape[0])
    dimensions.append(filedata.shape[1])
    if n_layers == 0:
        n_layers = filedata.shape[2]
    else:
        # Check that all images have same number of layers
        assert n_layers == filedata.shape[2]

min_dimension = min(dimensions)

#################################################################
# zero padding to make the images square
#################################################################
print("Zero padding and standardizing image sizes")
images = []
for filename in glob.glob(filepath):
    rgb = iio.v3.imread(filename)
    diff = max(rgb.shape[0:2]) - min(rgb.shape[0:2]) # find the number of zero pad
    zpad = int(diff/2)+1
    x, y = rgb.shape[1], rgb.shape[0]
    
    # decide which dimension to zero pad, and adjust if the dimension difference is even or odd
    if x<y:
        if (diff % 2) == 0:
            rgb = np.pad(rgb, pad_width=[(0, 0),(zpad, zpad),(0, 0)], mode="constant")
        else:
            rgb = np.pad(rgb, pad_width=[(1, 0),(zpad, zpad),(0, 0)], mode="constant")
    elif x>y:
        if (diff % 2) == 0:
            rgb = np.pad(rgb, pad_width=[(zpad, zpad),(0, 0),(0, 0)], mode="constant")
        else:
            rgb = np.pad(rgb, pad_width=[(zpad, zpad),(1, 0),(0, 0)], mode="constant")
    else:
        # Case if image is perfect square already... still need to convert to numpy
        # so everything comes out in same format
        rgb = np.array(rgb)
            
    ######### cut the images to the minimum dimension of the image in the folder
    ######### to make them all the same size.
    rgb_shape = rgb.shape[0]
    lower_cut = (rgb_shape - min_dimension)/2
    upper_cut = rgb_shape - (rgb_shape - min_dimension)/2

    rgb = rgb[int(lower_cut):int(upper_cut), int(lower_cut):int(upper_cut), ]

    images.append(rgb.flatten())

images = np.array(images)
'''

########## Initialize ##########
# read in image, flatten it, put it into an array
r = 500
c = 500
flat_images = []
pixels = []
for img in os.listdir(filepath):
    pic = cv2.imread(os.path.join(filepath, img))
    #pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (r, c))
    # flatten image
    pic = np.array(pic.reshape(r * c, 3))
    flat_img = pic.flatten()

    flat_images.append(flat_img)

images = np.array(flat_images)
'''


#####################################################
######## ISOMAP with 6 neighbors and reduce to 2 s (each image s reduces to 2)
#####################################################
print("Run Isomap")
iso = manifold.Isomap(n_neighbors=6, n_components=2)
iso.fit(images)
manifold_raw = iso.transform(images)
manifold = pd.DataFrame(manifold_raw)

# Left with 2 dimensions
manifold.head()

####################################
########### Plotting ###############
####################################
print("Plotting results of top 2 components")
if make_plots:
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    ax = fig.add_subplot(111)
    ax.set_title('Final 2D from Isomap of Animal Images')
    ax.set_xlabel('dimension: 0')
    ax.set_ylabel('dimension: 1')

    # plot Isomap results
    image_dimensions = (min_dimension, min_dimension, n_layers)
    x_size = (max(manifold[0]) - min(manifold[0])) * 0.12
    y_size = (max(manifold[1]) - min(manifold[1])) * 0.12

    for img_num in range(len(images)):
        x0 = manifold.loc[img_num, 0] - (x_size / 2.)
        y0 = manifold.loc[img_num, 1] - (y_size / 2.)
        x1 = manifold.loc[img_num, 0] + (x_size / 2.)
        y1 = manifold.loc[img_num, 1] + (y_size / 2.)
        img = images[img_num].reshape(image_dimensions)

        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
                  interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # Show 2D components plot
    ax.scatter(manifold[0], manifold[1], marker='.', alpha=0.7)

    plt.show()

############################################
#### Find closest image in vector space ####
############################################
# Maybe pick a few sample images, and then display / indicate which is the closest image

