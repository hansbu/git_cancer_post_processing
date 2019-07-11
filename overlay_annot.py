import numpy as np
import os
import sys
import cv2
import openslide

is_Seer = False

mask_fol = 'TCGA_masks'
svs_fol = 'TCGA_imgs_test'
overlay_fol = 'overlay_heatmap/tcga/'

if is_Seer:
    mask_fol = 'seer_masks_test'
    svs_fol = '/data01/shared/hanle/tumor_project/breast_cancer_40X/cancer_masks'
    overlay_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/overlay_heatmap/seer'



seer_samples = set(['TCGA-AR-A0TV-01Z-00-DX1'])

mask_files = [f for f in os.listdir(mask_fol) if len(f) > 5]

for count, fn in enumerate(mask_files):
    slide_id = fn.split('.')[0]
    #if slide_id not in seer_samples: continue

    print(count, fn)
    #if not os.path.exists(os.path.join(overlay_fol, slide_id + '_overlay.png')):
    #    print('file not exists: ', slide_id)
    #    continue

    #overlay = cv2.imread(os.path.join(overlay_fol, slide_id + '_overlay.png'))

    mask = cv2.imread(os.path.join(mask_fol, fn))
    #mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (int(mask.shape[1]/4), int(mask.shape[0]/4)), interpolation = cv2.INTER_AREA)

    if is_Seer:
        svs = cv2.imread(os.path.join(svs_fol, slide_id + '.png'))  # for seer
    else:
        svs = cv2.imread(os.path.join(svs_fol, fn))  # for TCGA

    svs = cv2.resize(svs, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)

    annot = cv2.Canny(mask, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    annot = cv2.dilate(annot, kernel, iterations=1)

    for row in range(annot.shape[0]):
        for col in range(annot.shape[1]):
            if annot[row][col] > 0:
                svs[row, col, :] = 0
                svs[row, col, 2] = 255

    #temp = np.hstack((mask, overlay))
    #cv2.imshow('img', svs)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join(overlay_fol, slide_id + '_annoted.png'), svs)
