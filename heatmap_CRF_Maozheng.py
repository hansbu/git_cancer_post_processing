import os
import sys
import cv2
import numpy as np
import multiprocessing as mp
from crf_refine import *
import pdb

# iml = crf_refine_(iml)
# iml = cv2.GaussianBlur(iml, (7, 7), 0)

parent = 'pred_out/'
# fns = [f for f in os.listdir(parent) if (f[-7:-4] is not 'ori') and (f[-11:-4] is not 'overlay')]

fns = [os.path.join(parent, f) for f in os.listdir(parent) if '.png' in f]
print(len(fns), fns)


os.system('rm ' + parent + '*crf.png')

for fn in fns:
    print(fn)
    img = cv2.imread(fn, 0)
    print(np.min(img), np.max(img))

    iml_refined = crf_refine_(img/255)
    iml_refined = (iml_refined/np.max(iml_refined)*np.max(img)).astype(np.uint8)

    # iml_refined = np.hstack((img, iml_refined))
    # iml_refined = cv2.applyColorMap(iml_refined, cv2.COLORMAP_JET)

    # iml_refined_blurred = cv2.GaussianBlur(iml_refined, (7, 7), 0)
    # img_save = np.vstack((iml_refined.astype(np.uint8), iml_refined_blurred.astype(np.uint8)))

    cv2.imwrite(fn[:-4] + '_crf.png', iml_refined)

