import os
import sys
import cv2
import openslide
import numpy as np


mask_fol = 'TCGA_masks'
pred_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Cancer_heatmap_tcga_seer_v2'
svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'
out_fol = 'pred_gts'
mask_files = [x for x in os.listdir(mask_fol) if 'DX' in x]


mask_fn = 'TCGA-BH-A1EV-01Z-00-DX1.106CF220-1D7D-40DD-88B2-A7F00B758F8F.png'

temp = ['TCGA-AR-A1AM-01Z-00-DX1.B3F006D9-9386-41E5-B0B1-B0832EE104A0.png']
slide_id = mask_fn.split('.')[0]


temp = [x.strip() for x in open('slides_map.txt', 'r')]
slides_map = {}
for t in temp:
    t = t.split()
    slides_map[t[0]] = int(t[1])

for count, mask_fn in enumerate(mask_files):
    slide_id = mask_fn.split('.')[0]

    mask_path = os.path.join(mask_fol, mask_fn)
    pred_path = os.path.join(pred_fol, 'prediction-' + slide_id + '.low_res')
    svs_path = os.path.join(svs_fol, slide_id + '.svs')

    oslide = openslide.OpenSlide(svs_path)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    preds = np.loadtxt(pred_path)
    gts = cv2.imread(mask_path, 0)
    temp = gts.copy()

    gts[temp < 100] = 0
    gts[temp >= 100] = 1

    print(count, slide_id)

    scale = height/gts.shape[0]
    s = int(170 / scale)
    pred_gts = np.zeros((preds.shape[0], 2))

    if not os.path.exists(out_fol): os.mkdir(out_fol)

    for ind, pred in enumerate(preds):
        x, y, pred = pred[0], pred[1], pred[2]

        x0, y0 = int(x/scale), int(y/scale)
        if y0-s < 0 or y0 + s >= gts.shape[0] or x0 - s < 0 or x0 + s >= gts.shape[1]:
            gt = 0
        else:
            patch = gts[y0-s:y0+s, x0-s:x0+s]
            gt = int(np.sum(patch)/(patch.shape[0]*patch.shape[1]) > 0.5)

        if slide_id in slides_map:
            if slides_map[slide_id] == 2:
                if x > width/2:
                    pred = 0
                    gt = 0
            elif slides_map[slide_id] == 3:
                if x < width/3 or x > 2*width/3:
                    pred = 0
                    gt = 0
            elif slides_map[slide_id] == 4:
                if x < width/4 or x > width/2:
                    pred = 0
                    gt = 0

        pred_gts[ind] = np.array([pred, gt])

    np.save(os.path.join(out_fol, slide_id + '_preds_gts'), pred_gts)
