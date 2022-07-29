#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 03:46:56 2022

@author: mojtabataghipourkaffash
"""
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn import manifold
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imageio as iio
import math
import pandas as pd
import scipy.io
import glob

import matplotlib.pyplot as plt
import numpy as np

#############################################################
# read a series of images in a folder and save them in a list
#############################################################

## Configuration
# filepath = "/home/spencervore/OneDrive/isye6740_project_data/pexel/green_medium/*.jpeg"
filepath = "/home/spencervore/OneDrive/isye6740_project_data/pexel/test_mojtaba/*.jpeg"
filepath = "/home/spencervore/OneDrive/isye6740_project_data/pixabay/green/*.jpg"

make_plots = True

#################################################################
# Find minimum dimension of the images (x or y dim) in the folder
#################################################################
print("Finding min dimensions")
dimensions = []
for filename in glob.glob(filepath):
    dimensions.append(iio.imread(filename).shape[0])
    dimensions.append(iio.imread(filename).shape[1])
min_dimension = min(dimensions)

#################################################################
# zero padding to make the images square
#################################################################
print("Zero padding and standardizing image sizes")
images = []
for filename in glob.glob(filepath):
    rgb = iio.imread(filename)
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

    '''
    if (lower_cut % 2) == 0:
        rgb = rgb[int(lower_cut):int(upper_cut)+1, int(lower_cut):int(upper_cut)+1, ]
    else:
        rgb = rgb[int(lower_cut):int(upper_cut), int(lower_cut):int(upper_cut), ]
    '''
    # I think the above was making this come out as two different sizes. For this to work, everything must be exactly
    # the same size no matter how it's processed.
    rgb = rgb[int(lower_cut):int(upper_cut), int(lower_cut):int(upper_cut), ]

    images.append(rgb.flatten())

images=np.array(images)
images_count = len(images)

'''
#################################################################
# flatten data and convert it to data frame
#################################################################    

print("Flattening image list")
lst = []
for i in range(images_count, 0, -1):
    # Loop backwards and delete each item as we come to it to keep down data size.
    lst.append(images.pop(i).flatten())

del images
'''

'''
pd.options.display.max_columns = 7
df = pd.DataFrame(images)
df.head()
'''


#####################################################
######## ISOMAP with 6 neighbors and reduce to 2 s (each image s reduces to 2)
#####################################################
print("Run Isomap")
iso = manifold.Isomap(n_neighbors=6, n_components=2)
iso.fit(images)
manifold_2Da = iso.transform(images)
manifold_2D = pd.DataFrame(manifold_2Da, columns=['dimension 1', 'dimension 2'])

# Left with 2 dimensions
manifold_2D.head()

####################################
########### Plotting ###############
####################################
print("Plotting results")
if make_plots:
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    ax = fig.add_subplot(111)
    ax.set_title('Fnial 2D from Isomap of Animal Images')
    ax.set_xlabel('dimension: 1')
    ax.set_ylabel('dimension: 2')

    # plot Isomap results
    num_images = images_count
    image_dimensions = (min_dimension, min_dimension, 3)
    x_size = (max(manifold_2D['dimension 1']) - min(manifold_2D['dimension 1'])) * 0.12
    y_size = (max(manifold_2D['dimension 2']) - min(manifold_2D['dimension 2'])) * 0.12
    # for i in range(40):
    #     img_num = np.random.randint(0, num_images)
    #     x0 = manifold_2D.loc[img_num, 'dimension 1'] - (x_size / 2.)
    #     y0 = manifold_2D.loc[img_num, 'dimension 2'] - (y_size / 2.)
    #     x1 = manifold_2D.loc[img_num, 'dimension 1'] + (x_size / 2.)
    #     y1 = manifold_2D.loc[img_num, 'dimension 2'] + (y_size / 2.)
    #     img = lst[img_num].reshape(image_dimensions)
    #     ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
    #               interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    for img_num in range(len(images)):
        x0 = manifold_2D.loc[img_num, 'dimension 1'] - (x_size / 2.)
        y0 = manifold_2D.loc[img_num, 'dimension 2'] - (y_size / 2.)
        x1 = manifold_2D.loc[img_num, 'dimension 1'] + (x_size / 2.)
        y1 = manifold_2D.loc[img_num, 'dimension 2'] + (y_size / 2.)
        img = images[img_num].reshape(image_dimensions)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
                  interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # Show 2D components plot
    ax.scatter(manifold_2D['dimension 1'], manifold_2D['dimension 2'], marker='.',alpha=0.7)

    plt.show()

############################################
#### Find closest image in vector space ####
############################################
# Maybe pick a few sample images, and then display / indicate which is the closest image

