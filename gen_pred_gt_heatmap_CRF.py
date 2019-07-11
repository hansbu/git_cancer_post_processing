import os
import sys
import cv2
import openslide
import numpy as np
from dice_auc_cal import *
import multiprocessing as mp

print('Usage: python gen_pred_gt_heatmap.py folder_to_predition_file extension_of_pred_file')
print('For example: \npred_fol = /data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v1 \nextension = low_res_xxxxx\n')


pred_fol = sys.argv[1]
extension = sys.argv[2]
out_fol = os.path.join(pred_fol, extension)

if extension == '':
    out_fol = os.path.join(pred_fol, 'high_res')

print('========================')
print('pred_fol: ', pred_fol)
print('pred extension: ', extension)

mask_fol = 'seer_masks_test'
svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'

mask_files = [x for x in os.listdir(mask_fol) if len(x) > 5]

temp = [x.strip() for x in open('slides_map.txt', 'r')]
slides_map = {}
for t in temp:
    t = t.split()
    slides_map[t[0]] = int(t[1])

if extension != '':
    extension = '.' + extension

for count, mask_fn in enumerate(mask_files):
    slide_id = mask_fn.split('.')[0]

    mask_path = os.path.join(mask_fol, mask_fn)

    pred_path = os.path.join(pred_fol, 'prediction-' + slide_id + extension)
    svs_path = os.path.join(svs_fol, slide_id + '.svs')

    oslide = openslide.OpenSlide(svs_path)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    preds = np.loadtxt(pred_path).astype(np.float32)
    gts = cv2.imread(mask_path, 0)
    if 'DX' not in mask_fn:
        gts = cv2.resize(gts, (0, 0), fx = 0.5, fy = 0.5)
    temp = gts.copy()

    gts[temp < 240] = 0
    gts[temp >= 240] = 1

    x = preds[:, 0]
    y = preds[:, 1]
    l = preds[:, 2]
    patch_size = (x.min() + x.max()) / len(np.unique(x))

    x = np.round((x + patch_size / 2.0) / patch_size)
    y = np.round((y + patch_size / 2.0) / patch_size)

    scale = height/gts.shape[0]
    up = int(patch_size/scale)

    iml = np.zeros((int(y.max()), int(x.max())), dtype=np.float32)

    # print('shape of iml: ', iml.shape)
    # print('len(x): ', len(x))

    for iter in range(len(x)):
        iml[int(y[iter] - 1), int(x[iter] - 1)] = l[iter]

    # apply CRF here!!!
    from crf_refine import *

    # iml = crf_refine_(iml)
    # iml = cv2.GaussianBlur(iml, (7, 7), 0)

    print(mask_fn)
    iml_refined = crf_refine_(iml)
    iml_refined = (iml_refined*255).astype(np.uint8)
    iml = (iml*255).astype(np.uint8)
    iml_refined = np.hstack((iml, iml_refined))
    iml_refined = cv2.applyColorMap(iml_refined, cv2.COLORMAP_JET)
    iml_refined_blurred = cv2.GaussianBlur(iml_refined, (7, 7), 0)
    cv2.imshow('img', np.vstack((iml_refined, iml_refined_blurred)))
    cv2.waitKey(0)

#
#     iml_u = np.zeros((iml.shape[0] * up, iml.shape[1] * up), dtype=np.float32)
#
#     if not os.path.exists(out_fol): os.mkdir(out_fol)
#     dest = os.path.join(out_fol, 'low_res_max_4_CRF_smooth7')
#
#     if not os.path.exists(dest): os.mkdir(dest)
#
#     for x in range(iml.shape[1]):
#         for y in range(iml.shape[0]):
#             iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x]
#
#             if slide_id in slides_map:
#                 if slides_map[slide_id] == 2:
#                     if x > iml.shape[1] / 2:
#                         iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] * 0
#                 elif slides_map[slide_id] == 3:
#                     if x < iml.shape[1] / 3 or x > 2 * iml.shape[1] / 3:
#                         iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] * 0
#                 elif slides_map[slide_id] == 4:
#                     if x < iml.shape[1] / 4 or x > iml.shape[1] / 2:
#                         iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] * 0
#
#     gts = cv2.resize(gts, (iml_u.shape[1], iml_u.shape[0]))
#     preds_gts = np.concatenate((iml_u.reshape(-1, 1), gts.reshape(-1, 1)), axis=1)
#     print(count + 1, slide_id, preds_gts.shape)
#     np.save(os.path.join(dest, slide_id + '_preds_gts'), preds_gts)
#
# compute_dice(dest)