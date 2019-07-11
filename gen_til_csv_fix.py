import numpy as np
import openslide
import sys
import os
import random
from PIL import Image
import cv2

svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'

TIL_thresholded = 'TIL_thresholded/'
fns = [f for f in os.listdir(TIL_thresholded) if '.png' in f]
out_fol = 'TIL_thresholded_csv/'

TIL_thresholded = 'cancer_thresholded_ours/'
fns = [f for f in os.listdir(TIL_thresholded) if '.png' in f]
out_fol = 'cancer_thresholded_ours_csv/'

if not os.path.exists(out_fol): os.mkdir(out_fol)


for ind, fn in enumerate(fns):
    print(ind, fn)
    slide_ID = fn.split('.png')[0]
    #til_map = np.array(Image.open(TIL_thresholded + slide_ID + '.png').convert('RGB'))

    til_map = cv2.imread(os.path.join(TIL_thresholded, slide_ID + '.png'), 0)

    pw_20X = 175
    if not os.path.exists(os.path.join(svs_fol, slide_ID + '.svs')):
        print('File not exist: ', slide_ID + '.svs')
        continue

    oslide = openslide.OpenSlide(os.path.join(svs_fol, slide_ID + '.svs'))
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"])
    else:
        mag = 10.0 / float(0.254)

    pw = float(int(10 * pw_20X * mag / 20)) / 10.0
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]
    res_file = os.path.join(out_fol, slide_ID + '.txt')
    if os.path.exists(res_file): continue

    with open(res_file, 'w') as f:
        for pat_x in range(til_map.shape[1]):
            for pat_y in range(til_map.shape[0]):
                til_pos = til_map[pat_y, pat_x] / 255   # red channel
                #til_neg = til_map[pat_y, pat_x, 2] / 255   # blue channel
                x = int(pw * (pat_x + 1) - pw / 2.0) + 1
                y = int(pw * (pat_y + 1) - pw / 2.0) + 1
                #f.writelines('{} {} {} {}\n'.format(x, y, til_pos, til_neg))
                f.writelines('{} {} {}\n'.format(x, y, til_pos))

