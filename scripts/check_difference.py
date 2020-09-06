import glob2
import os
import cv2
import numpy as np


# Simple script to assess simulations are identical

db_left = 'data/output'
db_right = 'data/output2'

p_left = [p[len(db_left)+1:] for p in glob2.glob(os.path.join(db_left, '**', '*.png'))]
p_right = [p[len(db_right)+1:] for p in glob2.glob(os.path.join(db_right, '**', '*.png'))]

left_only, right_only, identical, different = [], [], [], []
different_imgs = np.zeros((0, 2))

for i, p in enumerate(p_left):
    print("{}/{}".format(i, len(p_left)))
    if p not in p_right:
        left_only.append(p)
    else:
        im_left = cv2.imread(os.path.join(db_left, p))
        im_right = cv2.imread(os.path.join(db_right, p))

        if np.all(im_left==im_right):
            identical.append(p)
        else:
            different.append(p)

            diff = np.abs(np.array(im_left, dtype=np.int) - np.array(im_right, dtype=np.int))
            different_imgs = np.vstack([different_imgs, [diff.mean(), diff.std()]])

for p in p_right:
    if p not in p_left:
        right_only.append(p)

print("left_only: ", len(left_only))
print("right_only: ", len(right_only))
print("identical: ", len(identical))
print("different: ", len(different))

print("images differences average: mean ", different_imgs[:, 0].mean(), 'std ', different_imgs[:, 1].mean())
print("images differences max: mean ", different_imgs[:, 0].max(), 'std ', different_imgs[:, 1].max())
print("NOTE: small differences might just relate to float rounding issues")

if different_imgs.shape[0] > 0:
    mask = different_imgs[:, 0] == different_imgs[:, 0].max()
    idx = different_imgs[:, 0].argsort().astype(int)
    print("Top 5 diff image:\n", np.hstack([np.array(different)[idx][-5:][::-1].reshape((-1, 1)), different_imgs[idx][-5:][::-1, :].reshape((-1, 2))]))