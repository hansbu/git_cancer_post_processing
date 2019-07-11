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
files = set(files)

dest = source + '/arxiv'
if not os.path.exists(dest):
    os.mkdir(dest)
svs = os.listdir(source)

c = 0
for i, f in enumerate(svs):
    if f.split('.')[0] in files: continue
    c += 1
    print(c)
    print(f)
    os.system('mv ' + source + '/*' + f + '* ' + dest)        # this is for color-

