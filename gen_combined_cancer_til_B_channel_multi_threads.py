import os
import cv2
import numpy as np
import math
import sys
from PIL import Image
import multiprocessing as mp

print('Usage: python gen_combined_cancer_til.py folder_cancer_png folder_lym_png')

# uncomment these 3 lines to enable input arguments
#cancer_fol = sys.argv[1]
#lym_fol  = sys.argv[2]
#out_fol = sys.argv[3]

cancer_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/heatmap_png_til_cancer_tumor/brca_cancer_png'
lym_fol = '/data10/shared/hanle/lym_data_for_publication/TIL_maps_after_thres_v1/brca'
out_fol = sys.argv[1]

combined_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/heatmap_png_til_cancer_tumor/til_cancer_png'

out_fol = os.path.join(out_fol)
cancer_fns = [f for f in os.listdir(cancer_fol) if '.png' in f]
#for count, fn in enumerate(cancer_fns):
def red_yellow_map(fn):
    slide_id = fn.split('.')[0].split('_cancer')[0]
    cancer_path = os.path.join(cancer_fol, fn)
    lym_path = os.path.join(lym_fol, slide_id + '.png')
    combined_path = os.path.join(combined_fol, slide_id + '.png')
    if not os.path.exists(cancer_path):
        print('Cancer not exists: ', cancer_path)
        return
    if  not os.path.exists(lym_path):
        print('Lym not exists: ', lym_path)
        return
    if not os.path.exists(combined_path):
        print('Combined not exists: ', combined_path)
        return

    cancer_img = cv2.imread(cancer_path)[:,:,2]
    X = cancer_img.copy()
    lym_img  = cv2.imread(lym_path)
    combined_img = cv2.imread(combined_path)

    combined_img = cv2.resize(combined_img, (lym_img.shape[1], lym_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    print(lym_img.shape, combined_img.shape)

    up = int(math.ceil(lym_img.shape[0]/cancer_img.shape[0]))
    print(fn,up)

    if up > 1:
        iml_u = np.zeros((cancer_img.shape[0] * up, cancer_img.shape[1] * up), dtype=np.float32)
        for x in range(cancer_img.shape[1]):
            for y in range(cancer_img.shape[0]):
                iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancer_img[y, x]
        cancer_img = iml_u.copy()

    smooth5 = cancer_img.astype(np.uint8)
    if np.max(cancer_img) < 2:
        smooth5 = (cancer_img*255).astype(np.uint8)
    smooth5 = cv2.resize(smooth5, (lym_img.shape[1], lym_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    smooth5 = cv2.GaussianBlur(smooth5, (5, 5), 0)
    alpha = smooth5*0
    for i in range(lym_img.shape[0]):
        for j in range(lym_img.shape[1]):
            b = lym_img[i, j, 0]; g = lym_img[i, j, 1]; r = lym_img[i, j, 2]
            is_tumor = smooth5[i, j] > 100
            is_lym = r > 100
            alpha[i, j] = 255                     # tissue by default
            if is_tumor and is_lym:               # both cancer and lym
                alpha[i, j] = 252
            elif r < 10 and g < 10 and b < 10:    # this is background
                alpha[i, j] = 0
            elif (not is_tumor) and is_lym:       # this is lym only
                alpha[i, j] = 253
            elif is_tumor and (not is_lym):       # this is cancer only
                alpha[i, j] = 254

    if not os.path.exists(out_fol): os.mkdir(out_fol)

    combined_img[:,:,0] = alpha
    cv2.imwrite(os.path.join(out_fol, slide_id + '.png'), combined_img)

pool = mp.Pool(processes=10)
pool.map(red_yellow_map, cancer_fns)
