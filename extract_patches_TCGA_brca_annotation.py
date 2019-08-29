import os
import sys
import numpy as np
import cv2
import openslide
from multiprocessing import Pool

pngs_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/brca_cancer_til_pathDB_aligned'
svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'

pngs_fns = [f for f in os.listdir(pngs_fol) if '.png' in f]


#for png in pngs_fns[:1]:
def extract_position(png):
    slide_ID = png[:-4]
    oslide = openslide.OpenSlide(os.path.join(svs_fol, slide_ID + '.svs'))
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]
    png_img = cv2.imread(os.path.join(pngs_fol, png))

    cancer = png_img[:, :, 1]
    cancer[cancer < 128] = 0
    cancer[cancer >= 128] = 1
    cancer = cancer.astype(np.bool)
    lymph = png_img[:, :, 2].astype(np.bool)
    mask = np.logical_and(cancer, lymph)

    max_sum = 0
    ans = (0, 0)
    for r in range(mask.shape[0] - 20):
        for c in range(mask.shape[1] - 20):
            tmp = np.sum(mask[r:r + 20, c:c + 20])
            if max_sum < tmp:
                max_sum = tmp
                ans = (c, r)    # (x, y)
    ans = (slide_ID, int(ans[0]/mask.shape[1]*width), int(ans[1]/mask.shape[0]*height))
    print(png, ans)
    return ans

p = Pool(processes=10)
data = p.map(extract_position, pngs_fns)
p.close()
print(data)
f = open(os.path.join('TCGA_BRCA_finegrain_patches', 'TCGA_BRCA_finegrain_patches_list.txt'), 'w')
for slide_ID, x, y in data:
    f.writelines('{} {} {} {}\n'.format(slide_ID, x, y, 0))
f.close()


