import os
import sys
import glob
import numpy as np
import multiprocessing as mp

in_fol = '../heatmap_txt'
out_fol = '../heatmap_txt_separate_class/heatmap_txt_thresholded'
if not os.path.exists(out_fol):
    os.mkdir(out_fol)

probs = [0.3, 0.5, 0.7, 0.9]
files = glob.glob(in_fol + '/prediction*')

def process(file):
    print(file)
    slide_id = file.split('/')[-1]
    preds = [f.rstrip().split(' ') for f in open(file, 'r')]
    out = open(os.path.join(out_fol, slide_id), 'w')
    for pred in preds[1:]:
        grades = np.array([float(p) for p in pred[2:]])
        res = probs[np.argmax(grades)] if sum(grades) > 0 else 0
        out.writelines('{} {} {} 0 \n'.format(pred[0], pred[1], res))

    os.system('cp ' + os.path.join(in_fol, 'color-' + slide_id[11:]) + ' ' + os.path.join(out_fol,  'color-' + slide_id[11:]))
    out.close()

print(len(files))
pool = mp.Pool(processes=20)
pool.map(process, files)



