import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# set path to directory with pics
#path = r"C:\Users\lesea\Georgia Institute of Technology\Vore, Spencer E - isye6740_project_data\pixabay\test_transparent"
path = r"C:\Users\lesea\Georgia Institute of Technology\Vore, Spencer E - isye6740_project_data\test"

r = 1000
c = 1000
pics = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (r, c))
    # flatten image
    rows = np.shape(pic)[0]
    cols = np.shape(pic)[1]
    pic = pic.reshape(rows*cols)
    pics.append(pic)

pic_array = np.array(pics)

#####################################################
########### Running PCA ###############
#####################################################
# do PCA with 2 PCs
print("running PCA")
pca = PCA(n_components=2)
pca_imgs = pca.fit_transform(pic_array)

x = np.array(pca_imgs[:, 0])
y = np.array(pca_imgs[:, 1])

#####################################################
########### Outliers ###############
#####################################################
print("Removing Outliers")
# find outliers
x_outlier = np.where(x > 100000)[0].tolist()
y_outlier = np.where(y > 50000)[0].tolist()
outliers = x_outlier + y_outlier
outliers = np.unique(np.array(outliers))

# remove outliers
pca_imgs = np.delete(pca_imgs, outliers, 0)
pic_array = np.delete(pic_array, outliers, 0)

# redefine x and y
x = np.array(pca_imgs[:, 0])
y = np.array(pca_imgs[:, 1])

#####################################################
########### Plot ###############
#####################################################
# plot results
print("Plotting results")
fig, ax = plt.subplots()
plt.title('2D Plot Using PCA of Grayscale Animal Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

x_size = (max(x) - min(x))
y_size = (max(y) - min(y))
size = min(x_size, y_size)
x_size = size * .1
y_size = size * .1


pic = 0
for p in pca_imgs:
    x0 = p[0] - (x_size / 2)
    x1 = p[0] + (x_size / 2)
    y0 = p[1] - (y_size / 2)
    y1 = p[1] + (y_size / 2)

    img = pic_array[pic].reshape(r, c)
    ax.imshow(img, aspect='auto', cmap='gray',
              interpolation='nearest', extent=(x0, x1, y0, y1))

    ax.scatter(x, y, marker="None", alpha=0.7)

    pic += 1

plt.savefig('C:/Users/lesea/OneDrive/Documents/GA Tech/Summer 2022/ISYE 6740/Project/pca_grayscale.png')
plt.show()