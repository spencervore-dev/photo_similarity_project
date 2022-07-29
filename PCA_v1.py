import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# set path to directory with pics
path = "/home/spencervore/OneDrive/isye6740_project_data/pexel/green_medium"
#path = r"C:\Users\lesea\Georgia Institute of Technology\Vore, Spencer E - isye6740_project_data\pexel\test_mojtaba"

r = 500
c = 500
pics = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (r, c))
    # flatten image
    rows = np.shape(pic)[0]
    cols = np.shape(pic)[1]
    pic = pic.reshape(rows*cols)
#    print(np.shape(pic))
    pics.append(pic)

pic_array = np.array(pics)

# test showing image
# pictures = []
# for img in pic_array:
#     img = img.reshape(rows, cols)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     pictures.append(img)

# do PCA with 2 PCs
pca = PCA(n_components=2)
pca_imgs = pca.fit_transform(pic_array)

x = list(pca_imgs[0])

y = list(pca_imgs[1])

fig, ax = plt.subplots()
plt.title('2D Plot Using PCA of Grayscale Animal Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

x_size = (max(x) - min(x)) * 0.6
y_size = (max(y) - min(y)) * 0.6

pic = 0
for p in pca_imgs:
    x0 = p[0] - (x_size / 2)
    x1 = p[0] + (x_size / 2)
    y0 = p[1] - (y_size / 2)
    y1 = p[1] + (y_size / 2)

    img = pic_array[pic].reshape(r, c)
    ax.imshow(img, aspect='auto', cmap='gray',
              interpolation='nearest', extent=(x0, x1, y0, y1))

    ax.scatter(x, y, marker='.', alpha=0.7)

    pic += 1

plt.savefig('pca_v1.png')
plt.show()