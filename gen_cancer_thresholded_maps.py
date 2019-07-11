import os
import sys
import cv2
import openslide
import numpy as np
from dice_auc_cal import *
import multiprocessing as mp
import pickle

print('Usage: python gen_pred_gt_heatmap.py folder_to_predition_file extension_of_pred_file')
print('For example: \npred_fol = /data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v1 \nextension = low_res_xxxxx\n')

pred_fol = sys.argv[1]
# extension = sys.argv[2]

#pred_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v1'

extension = 'low_res'
til_thresholded = 'TIL_thresholded_csv'

til_cancer_fol = 'TIL_cancer_csv_ours'
cancer_thresholded = 'cancer_thresholded_ours'

til_cancer_fol = 'TIL_cancer_csv_HASHI'
cancer_thresholded = 'cancer_thresholded_HASHI'

if not os.path.exists(til_cancer_fol): os.mkdir(til_cancer_fol)
if not os.path.exists(cancer_thresholded): os.mkdir(cancer_thresholded)

out_fol = os.path.join(pred_fol, extension)

if extension == '':
    out_fol = os.path.join(pred_fol, 'high_res')

print('========================')
print('pred_fol: ', pred_fol)
print('pred extension: ', extension)

svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'
preds_files = [x for x in os.listdir(pred_fol) if 'prediction' in x and extension in x and 'TCGA' in x]
preds_files = list(set(preds_files))

# print(preds_files)

if extension != '':
    extension = '.' + extension

for count, pred_fn in enumerate(preds_files):
    slide_id = pred_fn.split('prediction-')[1].split('.')[0]

    # slide_id = 'TCGA-EW-A1PA-01Z-00-DX1'
    # pred_fn = 'prediction-' + slide_id + extension

    pred_path = os.path.join(pred_fol, pred_fn)
    svs_path = os.path.join(svs_fol, slide_id + '.svs')

    print(count, slide_id)

    if not os.path.exists(pred_path):
        print('\tcancer pred file not exists: ', pred_path)
        continue

    oslide = openslide.OpenSlide(svs_path)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    preds = np.loadtxt(pred_path).astype(np.float32)

    if not os.path.exists(os.path.join(til_thresholded, slide_id + '.txt')):
        print('\tTIL pred file not exists: ', slide_id)
        continue

    res_file = os.path.join(til_cancer_fol, slide_id + '.csv')
    if os.path.exists(res_file):
        print('\tfile generated')
        continue

    tils = np.loadtxt(os.path.join(til_thresholded, slide_id + '.txt'))

    x = preds[:, 0]
    y = preds[:, 1]
    l = preds[:, 2]
    patch_size = (x.min() + x.max()) / len(np.unique(x))
    patch_size_y = (y.min() + y.max()) / len(np.unique(y))

    # print('patch_size: ', patch_size, patch_size_y)
    # print('x_max, y_max: ', x.max(), y.max())

    x = np.round((x + patch_size / 2.0) / patch_size)
    y = np.round((y + patch_size / 2.0) / patch_size)

    iml = np.zeros((int(y.max()), int(x.max())), dtype=np.float32)

    for iter in range(len(x)):
        iml[int(y[iter] - 1), int(x[iter] - 1)] = l[iter]

    iml = cv2.GaussianBlur(iml, (7, 7), 0)

    print('\tshape of smooth7: ', iml.shape)

    up = int(min(width/iml.shape[1], height/iml.shape[0]))
    if up > 2000: up = int(up/2)
    #up = 4
    print('\tup: ', up)

    sub_fol = 'smooth7'
    iml_u = np.zeros((iml.shape[0] * up, iml.shape[1] * up), dtype=np.float32)

    if not os.path.exists(out_fol): os.mkdir(out_fol)
    dest = os.path.join(out_fol, sub_fol)
    if not os.path.exists(dest): os.mkdir(dest)

    for x in range(iml.shape[1]):
        for y in range(iml.shape[0]):
            iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = int(iml[y, x] > 0.6)

    #vis = (iml_u * 255).astype(np.uint8)
    #print('\tvis shape: ', vis.shape)
    #cv2.imwrite(os.path.join(cancer_thresholded, slide_id + '.png'), vis)


    scale_w = width/iml_u.shape[1]
    scale_h = height/iml_u.shape[0]
    print('\tscale_w, scale_h: ', scale_w, scale_h)

    print('\tshape of tils: ', tils.shape)
    patch_til = max(abs(tils[1,1] - tils[0,1]), abs(tils[1,0] - tils[0,0]))
    patch_w_half, patch_h_half = int(patch_til / scale_w / 2), int(patch_til / scale_h / 2)

    print('\tpatch til size, window_half: ', patch_til, patch_w_half, patch_h_half)

    # res_cancer = tils[:,2]*0
    res = -1
    c_pos = 0; c_neg = 0; c_bou = 0
    with open(res_file, 'w') as f:
        for ind, til in enumerate(tils):
            x, y, _, _ = tuple(til)
            x, y, patch_w_half, patch_h_half = int(x/scale_w), int(y/scale_h), int(patch_til/scale_w/2), int(patch_til/scale_h/2)

            window = iml_u[y - patch_h_half:y + patch_h_half, x - patch_w_half: x + patch_w_half]
            window_sum = np.sum(window)
            delta = 0.0
            if window_sum <= window.shape[0]*window.shape[1]*delta:     # this is non-cancer patch
                #res_cancer[ind] = 0
                res = 0
                c_neg += 1
            elif window_sum >= window.shape[0]*window.shape[1]*(1 - delta):     # this is cancer patch
                #res_cancer[ind] = 1
                res = 1
                c_pos += 1
            else:
                #res_cancer[ind] = 2
                res = 2
                c_bou += 1
            f.writelines('{},{},{},{},{}\n'.format(int(til[0]), int(til[1]), int(til[2]), int(res), int(max(til[3], til[2], res))))
        print('\tpos, neg, bou: ', c_pos, c_neg, c_bou)




