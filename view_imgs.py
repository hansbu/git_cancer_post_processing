import os
import sys
import cv2
import numpy as np


mask_fol = 'TCGA_masks'
img_fol = 'TCGA_imgs_idx5'

files = [f for f in os.listdir(mask_fol) if '.png' in f]

for file in files:
    print(file)
    mask = cv2.imread(os.path.join(mask_fol, file))
    mask = cv2.resize(mask, (0, 0), fx = 0.2, fy = 0.2)

    img = cv2.imread(os.path.join(img_fol, file))
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

    dum = np.zeros((img.shape[0], 10, 3)).astype(np.uint8)
    x = np.hstack((img, dum, mask))

    cv2.imshow('img', x)
    cv2.waitKey(0)
