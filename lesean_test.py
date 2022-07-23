import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# set path to directory with pics
path = "C:/Users/lesea/Georgia Institute of Technology/Vore, Spencer E - isye6740_project_data/pexel/test_lesean"

pics = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (250, 250))
    # flatten image
    rows = np.shape(pic)[0]
    cols = np.shape(pic)[1]
    pic = pic.reshape(rows*cols)
#    print(np.shape(pic))
    pics.append(pic)

pic_array = np.array(pics)

# test showing image
for img in pic_array:
    img = img.reshape(rows, cols)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    plt.imshow(img)
    plt.show()

print(np.shape(pic_array))