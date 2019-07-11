import os
import sys
import cv2
import openslide
import numpy as np
from dice_auc_cal import *


print('Usage: python gen_pred_gt_heatmap.py folder_to_predition_file extension_of_pred_file')
print('For example: \npred_fol = /data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v1')
print('extension = low_res_xxxxx\n')

# pred_fol = sys.argv[1]
# extension = sys.argv[2]

# pred_fol = '/data04/shared/hanle/paad_prediction/heatmap_gen/patch-level-lym'
pred_fol = '/data04/shared/hanle/paad_prediction/data/heatmap_txt_ensemble'
extension = ''

print('========================')
print('pred_fol: ', pred_fol)
print('pred extension: ', extension)

is_Seer = False
mask_fol = '/data04/shared/hanle/tumor_paad.dum1/data/tumor_labeled_heatmaps'
svs_fol = '/data01/shared/hanle/svs_seer_pancrease'
svs_resized_fol = '/data04/shared/hanle/paad_prediction/data/resized_svs'

if is_Seer:
    mask_fol = 'seer_masks_test'
    svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'

mask_files = [x for x in os.listdir(mask_fol) if 'gupta' in x]
print('number of mask files: ', len(mask_files))

if extension != '':
    extension = '.' + extension

for count, mask_fn in enumerate(mask_files):
    slide_id = mask_fn.split('.')[0]

    mask_path = os.path.join(mask_fol, mask_fn)

    pred_path = os.path.join(pred_fol, 'prediction-' + slide_id + extension)
    if not os.path.exists(pred_path):
        print('pred file not found: ', pred_path)
        continue

    svs_path = os.path.join(svs_fol, slide_id + '.svs')

    oslide = openslide.OpenSlide(svs_path)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    preds = np.loadtxt(pred_path).astype(np.float32)
    gts = cv2.imread(mask_path, 0)
    if 'DX' not in mask_fn: # this is SEER slides
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
    up = 4

    iml = np.zeros((int(y.max()), int(x.max())), dtype=np.float32)

    for iter in range(len(x)):
        iml[int(y[iter] - 1), int(x[iter] - 1)] = l[iter]

    # iml_u = np.zeros((iml.shape[0] * up, iml.shape[1] * up), dtype=np.float32)
    # for x in range(iml.shape[1]):
    #     for y in range(iml.shape[0]):
    #         iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = iml[y, x]
    # iml = iml_u.copy()

    smooth5 = (iml*255).astype(np.uint8)
    smooth5 = cv2.resize(smooth5, (0, 0), fx=6, fy=6, interpolation=cv2.INTER_LINEAR)
    smooth5 = cv2.GaussianBlur(smooth5, (7, 7), 0)

    t = 0.3
    indx = smooth5 > t*255
    smooth5[smooth5 < t*255] = 0

    print(count + 1, mask_fn)
    print(indx.shape)

    thresholded = smooth5.copy()
    thresholded[smooth5 > 0] = 1
    gts = cv2.resize(gts, (thresholded.shape[1], thresholded.shape[0]), interpolation=cv2.INTER_AREA)

    thresholded_gts = (np.hstack((thresholded, gts))*255).astype(np.uint8)

    TP = thresholded*gts
    TN = (1 - thresholded)*(1 - gts)
    FP = thresholded*(1 - gts)
    FN = (1 - thresholded)*gts

    confusion = np.zeros((thresholded.shape[0], thresholded.shape[1], 3))
    confusion[:, :, 0] = TN
    confusion[:, :, 1] = TP
    confusion[:, :, 2] = FN

    for row in range(FP.shape[0]):
        for col in range(FP.shape[1]):
            if FP[row][col] > 0:
                confusion[row, col, :] = 1
                confusion[row, col, 0] = 0

    confusion = (confusion*255).astype(np.uint8)
    # cv2.imshow('img', confusion)
    # cv2.waitKey(0)

    smooth5 = cv2.applyColorMap(smooth5, cv2.COLORMAP_JET)
    # smooth5 = cv2.GaussianBlur(smooth5, (3, 3), 0)

    blue = smooth5[:,:,0] > 125
    blue2 = smooth5[:,:,0] < 130
    green = smooth5[:, :, 1] < 10
    red = smooth5[:, :, 2] < 10
    bg = np.logical_and(np.logical_and(np.logical_and(blue, green), red), blue2)

    img_wsi = os.path.join(svs_resized_fol, slide_id + '_resized.png')
    if is_Seer: img_wsi = os.path.join('/data01/shared/hanle/tumor_project/breast_cancer_40X/cancer_masks', slide_id + '.png')

    if not os.path.exists(img_wsi):
        print('svs resized image not exists: ', img_wsi)
        continue

    img_wsi = cv2.imread(img_wsi)
    img_wsi = cv2.resize(img_wsi, (smooth5.shape[1], smooth5.shape[0]), interpolation = cv2.INTER_AREA)

    smooth5[:, :, 0][bg] = img_wsi[:, :, 0][bg]
    smooth5[:, :, 1][bg] = img_wsi[:, :, 1][bg]
    smooth5[:, :, 2][bg] = img_wsi[:, :, 2][bg]

    combined = np.hstack((img_wsi, smooth5))

    dest = 'overlay_heatmap/paad/ensemble'

    if not os.path.exists(dest): os.mkdir(dest)
    cv2.imwrite(os.path.join(dest, slide_id + '_overlay.png'), smooth5)