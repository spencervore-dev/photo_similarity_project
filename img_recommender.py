from scipy import spatial
import cv2
import matplotlib.pyplot as plt


def img_recommender(isomap_imgs, flat_pics, indexes_to_recommend_for, savepath, n_components='undefined', r=500, c=500):
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
            plt.suptitle(f"Recommendation {k+1} for image #{ind}", fontsize=14)
            plt.title(f"Distance = {int(dists[k])}, n_components = {n_components}", fontsize=9)
            plt.axis("off")
            # plt.show()
            plt.savefig(savepath + f"original_no{ind}_n{n_components}_rec_no{k+1}.png")
            plt.clf()
    return