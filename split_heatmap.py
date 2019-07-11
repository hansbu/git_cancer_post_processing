import os
import sys
from shutil import copyfile as cp

#path = '/data01/shared/hanle/SEER_clustering/u24_lymphocyte_1/data/threshold_list.txt'
#path = '/data01/shared/hanle/SEER_clustering/u24_lymphocyte_9/data/heatmap_txt/node009.txt'

#source = '/data01/shared/hanle/SEER_clustering/heatmap_text_359_correct'
#source = '/data01/shared/hanle/SEER_clustering/u24_lymphocyte_9/data/heatmap_txt'

print('usage: python split_heatmap.py path_textfile_contain_slides source_folder')

path = sys.argv[1]
source = sys.argv[2]

files = [f.rstrip('\n').split()[0] for f in open(path)]
#files = set(files)

c = 0
for i, f in enumerate(files):
    c += 1
    print(c)
    dest = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/heatmap_txt_TIL_tcga/'
    #os.system('cp ' + source + '/prediction-' + f + '* ' + dest + 'prediction-' + f)        # this is high_res
    #os.system('cp ' + source + '/prediction-' + f + '* ' + dest + 'prediction-' + f + '.low_res')   # this is low_res
    os.system('cp ' + source + '/color-' + f + ' ' + dest + 'color-' + f)        # this is for color-

