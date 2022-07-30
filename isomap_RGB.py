from sklearn.decomposition import PCA
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


#############################################################
# read a series of images in a folder and save them in a list
#############################################################

## Configuration
path = "/home/spencervore/OneDrive/isye6740_project_data/transparent_images/"

### for some reason this one image does not work:
# "pixabay_image__pg0002_ind125_imgid7086605_animals__animal_feline_tiger"

########## Initialize ##########
# read in image, flatten it, put it into an array
r = 500
c = 500
flat_pics = []
pixels = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path, img))
    #pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (r, c))
    # flatten image
    pic = np.array(pic.reshape(r * c, 3))
    flat_pic = pic.flatten()

    flat_pics.append(flat_pic)

flat_pics = np.array(flat_pics)
#####################################################
######## PCA with 2 PCs
#####################################################
print("Run Isomap")
iso = manifold.Isomap(n_neighbors=6, n_components=2)
iso.fit(flat_pics)
isomap_imgs = iso.transform(flat_pics)
# manifold = pd.DataFrame(manifold_raw)

# Left with 2 dimensions
# manifold.head()

x = np.array(isomap_imgs[:, 0])
y = np.array(isomap_imgs[:, 1])

'''
print("Running PCA")
pca = PCA(n_components=2)
isomap_imgs = pca.fit_transform(flat_pics)

x = np.array(isomap_imgs[:, 0])
y = np.array(isomap_imgs[:, 1])
'''

#####################################################
########### Outliers ###############
#####################################################
print("Removing Outliers")
# find outliers
x_outlier = np.where(x > 75000)[0].tolist()
y_outlier = np.where(y > 40000)[0].tolist()
outliers = x_outlier + y_outlier
outliers = np.unique(np.array(outliers))

# remove outliers
isomap_imgs = np.delete(isomap_imgs, outliers, 0)
flat_pics = np.delete(flat_pics, outliers, 0)

# redefine x and y
x = np.array(isomap_imgs[:, 0])
y = np.array(isomap_imgs[:, 1])

#####################################################
########### Plotting ###############
#####################################################
print("Plotting results")

fig, ax = plt.subplots()
plt.title('2D Plot Using PCA of RGB Animal Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

x_size = (max(x) - min(x))
y_size = (max(y) - min(y))
size = min(x_size, y_size)
x_size = size * .075
y_size = size * .075

pic = 0
for p in isomap_imgs:
    x0 = p[0] - (x_size / 2)
    x1 = p[0] + (x_size / 2)
    y0 = p[1] - (y_size / 2)
    y1 = p[1] + (y_size / 2)

    img = flat_pics[pic].reshape(r, c, 3)

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=(x0, x1, y0, y1))
    ax.scatter(x, y, marker="None", alpha=0.7)

    pic += 1

plt.savefig('isomap_RGB.png')
plt.show()
