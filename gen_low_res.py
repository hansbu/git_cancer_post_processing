import os
import numpy as np
import math
import scipy.stats as ss
import sys
import cv2


directory = sys.argv[1]     # pred folder
s = int(sys.argv[2])

# type = int(sys.argv[3])  # 0 for max, 1 for average, 2 for median

type = 0
extension = '.low_res_max_' + str(s)

# if type == 0:
#     extension = '.low_res_max_' + str(s)
# elif type == 1:
#     extension = '.low_res_avg_' + str(s)
# elif type == 2:
#     extension = '.low_res_median_' + str(s)
# else:
#     print('choose a type of aggregation!')


def low_res(fpath):
    x = np.zeros((10000000,), np.uint32);
    y = np.zeros((10000000,), np.uint32);
    p = np.zeros((10000000,), np.float32);
    n = np.zeros((10000000,), np.float32);

    nline = 0;
    with open(fpath) as f:
        for line in f:
            fields = line.split();
            if (len(fields) == 4):
                x[nline] = int(fields[0]);
                y[nline] = int(fields[1]);
                p[nline] = float(fields[2]);
                n[nline] = float(fields[3]);
                nline += 1;
    x = x[0:nline];
    y = y[0:nline];
    p = p[0:nline];
    n = n[0:nline];

    max_x = np.max(x) + np.min(x);
    x = (ss.rankdata(x, method='dense') - 1).astype(np.uint32);
    y = (ss.rankdata(y, method='dense') - 1).astype(np.uint32);
    step = max_x / (np.max(x) + 1);

    imp = np.zeros((np.max(x)+1, np.max(y)+1), np.float32);
    for it in range(len(x)):
        imp[x[it], y[it]] = p[it];

#    imp = cv2.GaussianBlur(imp, (9, 9), 0)

    imn = np.zeros((np.max(x)+1, np.max(y)+1), np.float32);
    for it in range(len(x)):
        imn[x[it], y[it]] = n[it];

    f = open(fpath + extension, 'w');
    for i in range(int(imp.shape[0]/s)):
        for j in range(int(imp.shape[1]/s)):
            if type == 0:
                p_val = np.max(imp[i*s:i*s+s, j*s:j*s+s]);
            elif type == 1:
                p_val = np.mean(imp[i * s:i * s + s, j * s:j * s + s]);
            elif type == 2:
                p_val = np.median(imp[i * s:i * s + s, j * s:j * s + s]);
            else:
                print("choose a type of aggregation!")

            n_val = np.min(imn[i*s:i*s+s, j*s:j*s+s]);
            f.write('{} {} {} {}\n'.format( \
                int(round((i+0.5)*step*s)), int(round((j+0.5)*step*s)), round(p_val, 6), round(n_val, 6)));
    f.close();


# directory = './patch-level-merged/';
fns = [fn for fn in os.listdir(directory) if ('.low' not in fn) and ('prediction' in fn)]
for ind, fn in enumerate(fns):
    fpath = os.path.join(directory, fn)
    if not os.path.isfile(fpath):
        continue;
    print(ind + 1, fpath)
    low_res(fpath);
