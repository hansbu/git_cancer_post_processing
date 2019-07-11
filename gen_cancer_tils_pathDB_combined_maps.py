import os
import sys
import cv2
import openslide
import numpy as np
from dice_auc_cal import *
import multiprocessing as mp
import pickle
import gc

try:
    start_ind = int(sys.argv[1])
    end_ind = int(sys.argv[2])
except:
    start_ind = 0
    end_ind = 10000

# input the path to cancer heatmap_txt and TIL heatmap_txt. Change the folders here!
cancer_pred_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v1'    # folder path containing the prediction-xxx files for cancer
til_thresholded = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/TIL_heatmap_tcga'         # folder path containing the prediction-xxx files for TILs
svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'                           # folder path containing all the WSIs
slide_extension = '.svs'        # extension of the slide, can be .svs, .tiff, etc.
til_cancer_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/brca_cancer_til_pathDB'       # output folder
# done changing arguments

if not os.path.exists(til_cancer_fol): os.mkdir(til_cancer_fol)

cancer_preds_files = [x for x in os.listdir(cancer_pred_fol) if 'prediction-' in x and 'low_res' not in x]   # work on high res only
cancer_preds_files = list(set(cancer_preds_files))
print('Total number of files: ', len(cancer_preds_files))

cancer_preds_files.sort()
if end_ind > len(cancer_preds_files): end_ind = len(cancer_preds_files)
cancer_preds_files = cancer_preds_files[start_ind:end_ind]

# cancer_only = set(['TCGA-3C-AALI-01Z-00-DX1'])

def process_file(pred_fn):
    slide_id = pred_fn.split('prediction-')[1].split('.')[0]
    cancer_png_path = os.path.join(cancer_pred_fol, pred_fn)
    svs_path = os.path.join(svs_fol, slide_id + slide_extension)
    print(pred_fn)
    # if slide_id not in cancer_only: return

    if not os.path.exists(cancer_png_path):
        print('\tcancer pred file not exists: ', cancer_png_path)
        return

    if not os.path.exists(svs_path):
        print('\tsvs file not exists: ', svs_path)
        return

    oslide = openslide.OpenSlide(svs_path)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    if not os.path.exists(os.path.join(til_thresholded, 'prediction-' + slide_id)):
        print('\tTIL pred file not exists: ', slide_id)
        return
    res_file = os.path.join(til_cancer_fol, slide_id + '.csv')
    res_file_png = os.path.join(til_cancer_fol, slide_id + '.png')
    if os.path.exists(res_file):
        print('\tfile generated')
        return

    cancer = np.loadtxt(os.path.join(cancer_pred_fol, 'prediction-' + slide_id))
    patch_cancer = max(abs(cancer[1, 1] - cancer[0, 1]), abs(cancer[1, 0] - cancer[0, 0]))

    iml = np.zeros((int(height/patch_cancer), int(width/patch_cancer)))
    print('\tsize of iml: ', iml.shape)
    print('\tcancer patch: ', patch_cancer)

    x = cancer[:, 0]
    y = cancer[:, 1]
    l = cancer[:, 2]

    print('max x, width, max_y, height: ', np.max(x), width, np.max(y), height)

    x = np.round((x - patch_cancer / 2.0) / patch_cancer)
    y = np.round((y - patch_cancer / 2.0) / patch_cancer)

    print('inds: ', np.min(x), np.max(x), np.min(y), np.max(y))

    # iml = np.zeros((int(np.max(y)), int(np.max(x))))
    for iter in range(len(x)):
        row, col = int(y[iter] - 1), int(x[iter] - 1)
        if row >= iml.shape[0] or col >= iml.shape[1]: continue

        iml[row, col] = l[iter]



    iml = cv2.GaussianBlur(iml, (3, 3), 0)

    up = int(min(width/iml.shape[1], height/iml.shape[0]))
    if up > 2000: up = int(up/2)
    print('\tup: ', up)

    iml_u = np.zeros((iml.shape[0] * up, iml.shape[1] * up), dtype=np.bool)        # this step is very slow
    for x in range(iml.shape[1]):
        for y in range(iml.shape[0]):
            iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] > 0.2  # threshold for cancer

    scale_w = width/iml_u.shape[1]
    scale_h = height/iml_u.shape[0]
    print('\tscale: ', scale_w, scale_h)

    tils = np.loadtxt(os.path.join(til_thresholded, 'prediction-' + slide_id))
    print('\tshape of tils: ', tils.shape)
    patch_til = max(abs(tils[1,1] - tils[0,1]), abs(tils[1,0] - tils[0,0]))
    patch_w_half, patch_h_half = int(patch_til / scale_w / 2), int(patch_til / scale_h / 2)

    print('\tpatch til size, window_half: ', patch_til, patch_w_half, patch_h_half)

    colors = np.loadtxt(os.path.join(til_thresholded, 'color-' + slide_id))
    tissue = colors[:, 2]
    colors = None; cancer = None; gc.collect()

    x, y = tils[:, 0], tils[:, 1]
    x, y = np.round((x + patch_til/2)/patch_til), np.round((y + patch_til/2)/patch_til)
    x_inds, y_inds = x.astype(np.int32) - 1, y.astype(np.int32) - 1
    combined = np.zeros((int(np.max(y_inds)) + 1, int(np.max(x_inds)) + 1, 3))


    c_pos = 0; c_neg = 0; c_bou = 0
    with open(res_file, 'w') as f:
        for ind, til in enumerate(tils):
            x, y, _, _ = tuple(til)
            x, y, patch_w_half, patch_h_half = int(x/scale_w), int(y/scale_h), int(patch_til/scale_w/2), int(patch_til/scale_h/2)

            if y - patch_h_half < 0 or y + patch_h_half >= iml_u.shape[0] or x - patch_w_half < 0 or x + patch_w_half >= iml_u.shape[1]:
                res = 0
                c_neg += 1
                f.writelines('{},{},{},{},{}\n'.format(int(til[0]), int(til[1]), 0, 0, 0))
            else:
                window = iml_u[y - patch_h_half:y + patch_h_half, x - patch_w_half: x + patch_w_half]
                window_sum = np.sum(window)
                delta = 0.0
                if window_sum <= window.shape[0]*window.shape[1]*delta:     # this is non-cancer patch
                    res = 0
                    c_neg += 1
                elif window_sum >= window.shape[0]*window.shape[1]*(1 - delta):     # this is cancer patch
                    res = 1
                    c_pos += 1
                else:                                                       # this is patch on the boundary
                    res = 2
                    c_bou += 1
                f.writelines('{},{},{},{},{}\n'.format(int(til[0]), int(til[1]), int(til[2]), int(res), int(tissue[ind] > 12)))

                isTil = til[2] if til[3] < 0.5 else 0       # filter Necrosis
                combined[y_inds[ind], x_inds[ind], :] = np.array([tissue[ind], int(iml[int((y - patch_h_half)/up), int((x - patch_w_half)/up)]*255), int(isTil*255)])

        print('\tpos, neg, bou: ', c_pos, c_neg, c_bou)

    tissue = combined[:,:,0]
    tissue = cv2.GaussianBlur(tissue, (5, 5), 0)        # blurring the tissue channel
    tissue[tissue < 12] = 0
    tissue[tissue >= 12] = 255
    combined[:,:,0] = tissue
    cv2.imwrite(res_file_png, combined)


pool = mp.Pool(processes=8)
pool.map(process_file, cancer_preds_files)
