import os
import numpy as np
from glob import glob
import collections
from collections import Counter
import multiprocessing as mp

def count_TILs_cancer_tissue(fn):
    out = collections.defaultdict(int)
    try:
        data = np.loadtxt(fn, delimiter=',', dtype='int')
        tissues = Counter(list(data[:, 4]))
        tils = Counter(list(data[:, 2]))
        tils_cancer = Counter([str(til) + str(cancer) for (til, cancer) in data[:, 2:4]])
        out['tissue'] = tissues[1] + tissues[2]
        out['total_tils'] = tils[1]
        out['tils_in_cancer'] = tils_cancer['11']
        out['tils_on_boundary'] = tils_cancer['12']
        out['tils_not_in_cancer'] = tils_cancer['10']
        out['slideID'] = fn.split('/')[-1][:-4]
        return out
    except:
        print('Error: ', fn)
        return collections.defaultdict(int)

def write_results_to_file(fid, counters):
    slideID, tissue, total_tils, tils_in_cancer, tils_on_boundary, tils_not_in_cancer = counters['slideID'], counters['tissue'],\
                     counters['total_tils'], counters['tils_in_cancer'], counters['tils_on_boundary'], counters['tils_not_in_cancer']
    fid.writelines('{},{},{},{},{},{}\n'.format(slideID, tissue, total_tils, tils_in_cancer, tils_on_boundary, tils_not_in_cancer))


if __name__ == '__main__':
    fol = 'TIL_cancer_csv_ours'
    fns = glob(fol + '/*.csv')

    fid = open(os.path.join(fol, 'summary_TILs_cancer_tissue.csv'), 'w')
    fid.writelines('slideID,total_tissue,total_tils,tils_in_cancer,tils_on_boundary,tils_NOT_in_cancer\n')

    pool = mp.Pool(processes=64)
    counters = pool.map(count_TILs_cancer_tissue, fns)
    pool.close()
    pool.join()

    for counter in counters:
        write_results_to_file(fid, counter)


