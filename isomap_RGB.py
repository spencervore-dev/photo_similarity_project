from sklearn.decomposition import PCA
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from scipy import spatial

#############################################################
# read a series of images in a folder and save them in a list
#############################################################

## Configuration
path = "/home/spencervore/OneDrive/isye6740_project_data/pixabay/transparent"
images_to_rec = [0, 100, 61, 11, 413, 380]
rec_save_loc = "../im_recs/pixabay_transparent_iso/pixa-transparent-iso-"


# path = "/home/spencervore/OneDrive/isye6740_project_data/pexel/test_mojtaba"
# images_to_rec = [0, 3]
# rec_save_loc = "../im_recs/test_mojtaba_iso_"


n_components = 2
outlier_removal = False

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
iso = manifold.Isomap(n_neighbors=6, n_components=n_components)
iso.fit(flat_pics)
isomap_imgs = iso.transform(flat_pics)
# manifold = pd.DataFrame(manifold_raw)

# Left with 2 dimensions
# manifold.head()

x = np.array(isomap_imgs[:, 0])
y = np.array(isomap_imgs[:, 1])

#####################################################
########### Outliers ###############
#####################################################
if outlier_removal:
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
# plt.show()
plt.clf()
print(f"Flat pics shape {flat_pics.shape}")


####################
# Recommend similar images
#################

def img_recommender(isomap_imgs, flat_pics, indexes_to_recommend_for, savepath):
    for ind in indexes_to_recommend_for:
        img = flat_pics[ind].reshape(r, c, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.suptitle(f"Original Image #{ind}", fontsize=14)
        plt.axis("off")
        # plt.show()
        plt.savefig(savepath + f"original_no{ind}.png")
        plt.clf()
        dists, ind_rec_vec = spatial.KDTree(isomap_imgs).query(isomap_imgs[ind], range(2, 6))
        for k, ind_rec in enumerate(ind_rec_vec):
            img = flat_pics[ind_rec].reshape(r, c, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.suptitle(f"Recommendation {k+1}", fontsize=14)
            plt.title(f"Distance = {int(dists[k])}, n_components = {n_components}", fontsize=9)
            plt.axis("off")
            # plt.show()
            plt.savefig(savepath + f"original_no{ind}_n{n_components}_rec_no{k+1}.png")
            plt.clf()
    return

img_recommender(isomap_imgs, flat_pics, images_to_rec, rec_save_loc)