import os

fns = [f for f in os.listdir('TCGA_masks') if '.png' in f]

for fn in fns:
    print(fn)
    os.system('cp ' + os.path.join('TCGA_masks', fn) + ' ' + os.path.join('tcga_masks', fn.split('.')[0] + '_pos.png'))
