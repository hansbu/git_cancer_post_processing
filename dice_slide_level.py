from dice_auc_cal import compute_dice_final
import numpy as np
import os
import sys


source = sys.argv[1]
threshold = float(sys.argv[2])

compute_dice_final(source, threshold)