import os
import sys
import cv2
import openslide
import numpy as np
from dice_auc_cal import *

print('Usage: python gen_pred_gt_heatmap.py folder_to_predition_file extension_of_pred_file')
print('For example: \npred_fol = /data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v1 \nextension = low_res_xxxxx\n')


# pred_fol = sys.argv[1]
# extension = sys.argv[2]

pred_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/tumor_heatmap_tcga_195'
extension = sys.argv[1]


out_fol = os.path.join(pred_fol, extension)

if extension == '':
    out_fol = os.path.join(pred_fol, 'high_res')
if not os.path.exists(out_fol): os.mkdir(out_fol)

print('========================')
print('pred_fol: ', pred_fol)
print('pred extension: ', extension)

# mask_fol = 'seer_masks_test'
mask_fol = 'TCGA_masks'

svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca/tcga_KE'

mask_files = [x for x in os.listdir(mask_fol) if len(x) > 5]

# mask_fn = 'TCGA-BH-A1EV-01Z-00-DX1.106CF220-1D7D-40DD-88B2-A7F00B758F8F.png'
# temp = ['TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A.png']
# slide_id = mask_fn.split('.')[0]


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

    print(svs_path)

    if not os.path.exists(pred_path):
        print('pred file not exists: ', pred_path)
        continue

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

    for iter in range(len(x)):
        iml[int(y[iter] - 1), int(x[iter] - 1)] = l[iter]

    iml_u = np.zeros((iml.shape[0] * up, iml.shape[1] * up), dtype=np.float32)

    for x in range(iml.shape[1]):
        for y in range(iml.shape[0]):
            iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x]

            # if slide_id in slides_map:
            #     if slides_map[slide_id] == 2:
            #         if x > iml.shape[1] / 2:
            #             iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] * 0
            #     elif slides_map[slide_id] == 3:
            #         if x < iml.shape[1] / 3 or x > 2 * iml.shape[1] / 3:
            #             iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] * 0
            #     elif slides_map[slide_id] == 4:
            #         if x < iml.shape[1] / 4 or x > iml.shape[1] / 2:
            #             iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x] * 0

    gts = cv2.resize(gts, (iml_u.shape[1], iml_u.shape[0]))
    preds_gts = np.concatenate((iml_u.reshape(-1, 1), gts.reshape(-1, 1)), axis=1)
    print(count + 1, slide_id, preds_gts.shape)

    # res = []
    # for theshold in np.array(range(1, 100))/100:
    #     res.append(dice_score(iml_u, gts, theshold))
    # res = np.array(res)
    # print(np.max(res))
    # print(np.argmax(res))

    np.save(os.path.join(out_fol, slide_id + '_preds_gts'), preds_gts)

    # iml_u = (iml_u * 255).astype(np.uint8)
    # gts = cv2.resize(gts, (iml_u.shape[1], iml_u.shape[0]))
    # gts = (gts * 255).astype(np.uint8)
    # print(iml_u.shape)
    # print(gts.shape)
    # temp = np.hstack((iml_u, gts))
    # temp = cv2.resize(temp, (0, 0), fx = 0.2, fy = 0.2)
    # cv2.imshow('img', temp)
    # cv2.waitKey(0)

compute_dice(out_fol, steps=10)
