import numpy as np
import os
import sys

source = sys.argv[1]
fns = [f for f in os.listdir(source) if 'preds_gts' in f]

IoU = 0
for i, fn in enumerate(fns):
    fn_path = os.path.join(source, fn)
    d = np.load(fn_path)
    d[d < 0.5] = 0
    d[d >= 0.5] = 1
    d = d.astype(dtype=bool)
    pred = d[:, 0]
    gt = d[:, 1]

    overlap = pred*gt
    union = pred + gt
    IoU += overlap.sum()/float(union.sum())
    print(i, fn, IoU/(i + 1))

print('\n\nIoU of {}: {:.4f}'.format(source, IoU/len(fns)))

