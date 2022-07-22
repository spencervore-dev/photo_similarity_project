#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:16:01 2022

@author: mojtabataghipourkaffash
"""
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

filepath = "/Users/mojtabataghipourkaffash/Desktop/Courses/ISYE 6740/Final Project/GitHub/images/*.jpeg"

#################################################################
# Find minimum dimension of the images (x or y dim) in the folder
#################################################################
dimensions = []
for filename in glob.glob(filepath):
    dimensions.append(iio.imread(filename).shape[0])
    dimensions.append(iio.imread(filename).shape[1])
min_dimension = min(dimensions)

#################################################################
# zero padding to make the images square
#################################################################
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
            
    ######### cut the images to the minimum dimension of the image in the folder
    ######### to make them all the same size.
    rgb_shape = rgb.shape[0]
    lower_cut = (rgb_shape - min_dimension)/2
    upper_cut = rgb_shape - (rgb_shape - min_dimension)/2

    if (lower_cut % 2) == 0:
        rgb = rgb[int(lower_cut):int(upper_cut)+1, int(lower_cut):int(upper_cut)+1, ]
    else:
        rgb = rgb[int(lower_cut):int(upper_cut), int(lower_cut):int(upper_cut), ]

    images.append(rgb)

#################################################################
# flatten data and convert it to data frame
#################################################################    

lst = []
for i in range(len(images)):
    lst.append(images[i].flatten())
    
pd.options.display.max_columns = 7
df = pd.DataFrame(lst)
df.head()

#####################################################
######## ISOMAP with 6 neighbors and reduce to 2 s (each image s reduces to 2)
#####################################################

iso = manifold.Isomap(n_neighbors=6, n_components=2)
iso.fit(lst)
manifold_2Da = iso.transform(lst)
manifold_2D = pd.DataFrame(manifold_2Da, columns=['dimension 1', 'dimension 2'])

# Left with 2 dimensions
manifold_2D.head()

####################################
########### Plotting ###############
####################################

fig = plt.figure()
fig.set_size_inches(9, 9)
ax = fig.add_subplot(111)
ax.set_title('Fnial 2D from Isomap of Animal Images')
ax.set_xlabel('dimension: 1')
ax.set_ylabel('dimension: 2')

# plot Isomap results
num_images = len(images)
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

for img_num in range(len(lst)):
    x0 = manifold_2D.loc[img_num, 'dimension 1'] - (x_size / 2.)
    y0 = manifold_2D.loc[img_num, 'dimension 2'] - (y_size / 2.)
    x1 = manifold_2D.loc[img_num, 'dimension 1'] + (x_size / 2.)
    y1 = manifold_2D.loc[img_num, 'dimension 2'] + (y_size / 2.)
    img = lst[img_num].reshape(image_dimensions)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
              interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

# Show 2D components plot
ax.scatter(manifold_2D['dimension 1'], manifold_2D['dimension 2'], marker='.',alpha=0.7)

plt.show()