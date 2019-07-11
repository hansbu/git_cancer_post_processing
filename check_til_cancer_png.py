import numpy as np
import cv2
import os

source = '/home/han/Downloads/columbia'
fns = [f for f in os.listdir(os.path.join(source, 'png_cancer')) if '.png' in f]

for fn in fns:
    cancer_path = os.path.join(source, 'png_cancer', fn)
    til_path = os.path.join(source, 'png_TIL', fn)

    cancer = cv2.imread(cancer_path)
    til = cv2.imread(til_path)
    h, w = til.shape[:2]

    til[:5, :, :] = 255
    til[h-5:h, :, :] = 255
    til[:, :5, :] = 255
    til[:, w-5:w, :] = 255

    cancer = cv2.resize(cancer, (til.shape[1], til.shape[0]))
    cancer[:5, :, :] = 255
    cancer[h-5:h, :, :] = 255
    cancer[:, :5, :] = 255
    cancer[:, w-5:w, :] = 255

    merge = np.hstack((cancer, til))
    cv2.imshow('merge', merge)
    cv2.waitKey(0)
