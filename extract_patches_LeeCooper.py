import numpy as np
import openslide
import sys
import os
from PIL import Image
import datetime
import time
import cv2
from shutil import copyfile as cp
import multiprocessing as mp
import collections
import imagesize


svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'
mask_fol = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/leeCooper_labels'
output_folder = 'leeCooper_patches'
if not os.path.exists(output_folder): os.mkdir(output_folder)

mask_fns = [fn for fn in os.listdir(mask_fol) if '.png' in fn]
mask_fns = [fn.split('_') for fn in mask_fns]
mask_fns = [(fn[0][:-4], fn[1][4:], fn[2][4:], '_'.join(fn)) for fn in mask_fns]


svs_fns = [fn for fn in os.listdir(svs_fol) if '.svs' in fn]
svs_maps = {fn[:12]:fn for fn in svs_fns}


def extract_patch(data):
    slide_id_short, x_min, y_min, mask_file = data
    if slide_id_short not in svs_maps:
        print('=========================WSI not available: ', slide_id_short)
        return

    fname = '{}/{}_input.png'.format(output_folder, mask_file[:-5]);
    if os.path.exists(fname):
        return

    svs_path = os.path.join(svs_fol, svs_maps[slide_id_short])
    w, h = imagesize.get(os.path.join(mask_fol, mask_file))
    print('extracting patch: ', mask_file)

    try:
        oslide = openslide.OpenSlide(svs_path);
    except:
        print('{}: exception caught'.format(svs_path));
        return

    patch = oslide.read_region((int(x_min), int(y_min)), 0, (int(w), int(h)));
    patch.save(fname);


pool = mp.Pool(processes=8)
pool.map(extract_patch, mask_fns)
