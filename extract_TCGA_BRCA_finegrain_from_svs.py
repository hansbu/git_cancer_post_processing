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


svs_fol = '/data01/shared/hanle/svs_tcga_seer_brca'
patch_size_20X = 2000

start = time.time()
output_folder = 'TCGA_BRCA_finegrain_patches'
if not os.path.exists(output_folder):
    os.mkdir(output_folder);

slide_corrs = [f.rstrip() for f in open('/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/TCGA_BRCA_finegrain_patches/TCGA_BRCA_finegrain_patches_list.txt', 'r')]
print(slide_corrs)

def extract_svs(fn):
    slide, x, y, lb = fn.split()
    print(slide)
    try:
        oslide = openslide.OpenSlide(os.path.join(svs_fol, slide + '.svs'));
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:     # 'openslide.mpp-x'
            mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        elif "XResolution" in oslide.properties:
            mag = 10.0 / float(oslide.properties["XResolution"]);
        elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
            Xres = float(oslide.properties["tiff.XResolution"])
            if Xres < 10:
                mag = 10.0 / Xres;
            else:
                mag = 10.0 / (10000/Xres)       # SEER PRAD
        else:
            print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide);
            mag = 10.0 / float(0.254);
        pw = int(patch_size_20X * mag / 20);  # scale patch size from 20X to 'mag'

        width = oslide.dimensions[0];
        height = oslide.dimensions[1];
    except:
        print('{}: exception caught'.format(slide));
        exit(1);


    fname = '{}/{}_{}_{}_{}_{}.png'.format(output_folder, slide, x, y, pw, patch_size_20X)
    patch = oslide.read_region((int(x), int(y)), 0, (pw, pw));
    patch_arr = np.array(patch);
    if(patch_arr[:,:,3].max() == 0):
        return
    patch = patch.resize((int(patch_size_20X), int(patch_size_20X)), Image.ANTIALIAS);
    patch.save(fname);


pool = mp.Pool(processes=20)
pool.map(extract_svs, slide_corrs)
print('Elapsed time: ', (time.time() - start)/60.0)

