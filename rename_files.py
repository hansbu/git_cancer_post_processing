import os
import sys


source = '/data04/shared/hanle/paad_prediction/download_heatmap/get_grayscale_heatmaps/grayscale_heatmaps/'
dest = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_png/heatmap_png_til_cancer_tumor/paad_cancer_png/'
fns = [f for f in os.listdir(source) if len(f) > 3]
for fn in fns:
    print(fn)
    os.system('cp ' + source + fn + ' ' + dest + fn.split('.')[0] + '.png')
