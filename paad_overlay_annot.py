import numpy as np
import os
import sys
import cv2
import openslide

is_Seer = False

mask_fol = '/data04/shared/hanle/tumor_paad.dum1/data/tumor_labeled_heatmaps'
svs_fol = '/data04/shared/hanle/paad_prediction/data/resized_svs'
overlay_fol = 'overlay_heatmap/paad/'

mask_files = [f for f in os.listdir(mask_fol) if 'gupta' in f]
scale = 32
for count, fn in enumerate(mask_files):
    slide_id = fn.split('.')[0]

    print(count, slide_id)

    svs_fn = os.path.join(svs_fol, slide_id + '_resized.png')
    if not os.path.exists(svs_fn):
        print('svs file not exist!')
        continue

    mask = cv2.imread(os.path.join(mask_fol, fn))
    mask = cv2.resize(mask, (int(mask.shape[1]/scale), int(mask.shape[0]/scale)), interpolation = cv2.INTER_AREA)
    svs = cv2.imread(svs_fn)  # for seer

    svs = cv2.resize(svs, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)

    annot = cv2.Canny(mask, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    annot = cv2.dilate(annot, kernel, iterations=1)

    for row in range(annot.shape[0]):
        for col in range(annot.shape[1]):
            if annot[row][col] > 0:
                svs[row, col, :] = 0
                svs[row, col, 2] = 255

    cv2.imwrite(os.path.join(overlay_fol, slide_id + '_annoted.png'), svs)
