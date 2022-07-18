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

images = []
for count, filename in enumerate(glob.glob(filepath)):
    if iio.imread(filename).shape==(350, 525, 3):
        images.append(iio.imread(filename))
#         print(images[count].shape)

num_images = len(images)
image_s = (350, 525, 3)

#########################################################
###### flatten data and convert it to data frame ########
#########################################################

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
manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])

# Left with 2 s
manifold_2D.head()

####################################
########### Plotting ###############
####################################

fig = plt.figure()
fig.set_size_inches(9, 9)
ax = fig.add_subplot(111)
ax.set_title('2D Components from Isomap')
ax.set_xlabel('Component: 1')
ax.set_ylabel('Component: 2')

# plot Isomap results
x_size = (max(manifold_2D['Component 1']) - min(manifold_2D['Component 1'])) * 0.12
y_size = (max(manifold_2D['Component 2']) - min(manifold_2D['Component 2'])) * 0.12
for i in range(40):
    img_num = np.random.randint(0, num_images)
    x0 = manifold_2D.loc[img_num, 'Component 1'] - (x_size / 2.)
    y0 = manifold_2D.loc[img_num, 'Component 2'] - (y_size / 2.)
    x1 = manifold_2D.loc[img_num, 'Component 1'] + (x_size / 2.)
    y1 = manifold_2D.loc[img_num, 'Component 2'] + (y_size / 2.)
    img = lst[img_num].reshape(image_s)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
              interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

# Show 2D components plot
ax.scatter(manifold_2D['Component 1'], manifold_2D['Component 2'], marker='.',alpha=0.7)

plt.show()