import cv2
import os
import sys
import numpy as np

fol = 'train_mask_ext'
fns = [f for f in os.listdir(fol) if len(f) > 3]
# print(fns)
for i, fn in enumerate(fns):
	print(i, fn)
	mask = cv2.imread(os.path.join(fol, fn), 0)
	unique, counts = numpy.unique(mask, return_counts=True)
	print(i, unique)