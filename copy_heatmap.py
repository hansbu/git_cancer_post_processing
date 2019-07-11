import os
import sys
from shutil import copyfile as cp

source = sys.argv[1]
dest = sys.argv[2]

files = [f.strip() for f in open('test_slides_seer.txt', 'r') if len(f) > 5]
for f in files:
    print(f)
    #high = 'prediction-' + f
    high = f + '.dr.rajarsi.gupta_Tumor_Region.png'
    cp(os.path.join(source, high), os.path.join(dest, high))
