import os
import cv2
import numpy as np
import math

print('Usage: python gen_combined_cancer_til_aligned.py folder_cancer_png folder_lym_png out_fol')

# cancer_fol = sys.argv[1]
# lym_fol = sys.argv[2]
# out_fol = sys.argv[3]

cancer_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/heatmap_png_til_cancer_tumor/paad_cancer_png'
lym_fol = '/data10/shared/hanle/lym_data_for_publication/TIL_maps_after_thres_v1/paad'
out_fol = 'paad_TCGA_cancer_lym_combined'

out_fol = os.path.join(out_fol)

cancer_fns = [f for f in os.listdir(cancer_fol) if '.png' in f]

for count, fn in enumerate(cancer_fns):
    slide_id = fn.split('.')[0]
    cancer_path = os.path.join(cancer_fol, fn)
    lym_path = os.path.join(lym_fol, fn)

    if not os.path.exists(cancer_path):
        print('File not exists: ', cancer_path)
        continue
    if  not os.path.exists(lym_path):
        print('File not exists: ', lym_path)
        continue

    cancer_img = cv2.imread(cancer_path)
    lym_img  = cv2.imread(lym_path)

    up = int(math.ceil(lym_img.shape[0]/cancer_img.shape[0]))
    print(count, fn,up)

    iml_u = np.zeros((cancer_img.shape[0] * up, cancer_img.shape[1] * up), dtype=np.float32)
    for x in range(cancer_img.shape[1]):
        for y in range(cancer_img.shape[0]):
            iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancer_img[y, x, 2]
    cancer_img = iml_u.copy()

    smooth5 = (cancer_img*255).astype(np.uint8)
    smooth5 = cv2.resize(smooth5, (lym_img.shape[1], lym_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    smooth5 = cv2.GaussianBlur(smooth5, (5, 5), 0)

    # t = 0.2
    # indx = smooth5 > t*255
    # smooth5[smooth5 < t*255] = 0
    # smooth5 = cv2.applyColorMap(smooth5, cv2.COLORMAP_JET)

    # cancer map is smooth5, channel Red > 100
    out = lym_img*0

    for i in range(lym_img.shape[0]):
        for j in range(lym_img.shape[1]):
            b = lym_img[i, j, 0]; g = lym_img[i, j, 1]; r = lym_img[i, j, 2]
            out[i, j] = np.array([192,192,192])
            if r < 10 and g < 10 and b < 10:    # this is background
                out[i, j] = np.array([255, 255, 255])
            if smooth5[i, j] > 100 and b == 255:  # this is cancer
                out[i, j] = np.array([0,255,255])
            if r > 100:  # this is lymphocytes
                out[i, j] = np.array([0,0,255])

    if not os.path.exists(out_fol): os.mkdir(out_fol)
    cv2.imwrite(os.path.join(out_fol, slide_id + '.png'), out)
