import os
import cv2
import numpy as np
import math
import sys
import multiprocessing as mp


print('Usage: python gen_combined_cancer_til.py folder_cancer_png folder_lym_png')

# uncomment these 3 lines to enable input arguments
#cancer_fol = sys.argv[1]
#lym_fol  = sys.argv[2]
#out_fol = sys.argv[3]

# paad
cancer_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/heatmap_png_til_cancer_tumor/paad_cancer_png'
lym_fol = '/data10/shared/hanle/lym_data_for_publication/TIL_maps_after_thres_v1/paad'
out_fol = 'paad_TCGA_cancer_lym_combined'

# prad
cancer_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/heatmap_png_til_cancer_tumor/prad_tcga_sample'
lym_fol = '/data10/shared/hanle/lym_data_for_publication/TIL_maps_after_thres_v1/prad'
out_fol = 'out_red_yellow_png/prad_red_yellow'

out_fol = os.path.join(out_fol)
cancer_fns = [f for f in os.listdir(cancer_fol) if '.png' in f]
def red_yellow_map(fn):
#for count, fn in enumerate(cancer_fns):
    slide_id = fn.split('.')[0].split('_cancer')[0]
    cancer_path = os.path.join(cancer_fol, fn)
    lym_path = os.path.join(lym_fol, slide_id + '.png')

    if not os.path.exists(cancer_path):
        print('Cancer not exists: ', cancer_path)
        return
    if  not os.path.exists(lym_path):
        print('Lym not exists: ', lym_path)
        return

    cancer_img = cv2.imread(cancer_path)[:,:,2]
    X = cancer_img.copy()
    lym_img  = cv2.imread(lym_path)

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
    out = lym_img*0
    for i in range(lym_img.shape[0]):
        for j in range(lym_img.shape[1]):
            b = lym_img[i, j, 0]; g = lym_img[i, j, 1]; r = lym_img[i, j, 2]
            out[i, j] = np.array([192,192,192])
            is_tumor = smooth5[i, j] > 100
            is_lym = r > 100
            if is_tumor and is_lym:       # both cancer and lym
                out[i, j] = np.array([0,0,255])
            elif r < 10 and g < 10 and b < 10:    # this is background
                out[i, j] = np.array([255, 255, 255])
            elif (not is_tumor) and is_lym:  # this is lym only
                out[i, j] = np.array([0,0,200])
            elif is_tumor and (not is_lym):  # this is cancer only
                out[i, j] = np.array([0,255,255])

    if not os.path.exists(out_fol): os.mkdir(out_fol)
    cv2.imwrite(os.path.join(out_fol, slide_id + '.png'), out)

pool = mp.Pool(processes=10)
pool.map(red_yellow_map, cancer_fns)

