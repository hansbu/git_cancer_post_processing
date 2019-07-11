import os

fns = [f.split('.')[0] for f in os.listdir('/data01/shared/hanle/svs_tcga_seer_brca/tcga_KE') if 'svs' in f]

for fn in fns:
    source = '/data01/shared/hanle/tumor_project/pub_tumor_cancer_brca/Tumor_heatmap_tcga'
    print(fn)
    os.system('cp ' + os.path.join(source, 'prediction-' + fn) + '* .')
