import os
import numpy as np

def load_txt(fn):
    data = np.loadtxt(fn)
    maps = {}
    for row in data:
        key = str(row[0]) + '_' + str(row[1])
        maps[key] = row[2:]
    return maps

def is_path_exists(fn):
    return os.path.exists(fn)

def process_fol(fol, out_fol):
    fn_stage1 = os.path.join(fol, 'patch-level-stage1.txt')
    fn_stage2 = os.path.join(fol, 'patch-level-stage2.txt')

    if not is_path_exists(fn_stage1) or not is_path_exists(fn_stage1):
        print('file not exits: ', fol)
        return

    pred_stage1 = load_txt(fn_stage1)
    pred_stage2 = load_txt(fn_stage2)
    prediction_fn = 'prediction-' + fol.split('/')[-1].split('.')[0]
    fid = open(os.path.join(out_fol, prediction_fn), 'w')
    fid.writelines('{} {} {} {} {} {}\n'.format('x_loc', 'y_loc', 'benign', 'grade3', 'grade4', 'grade5'))
    for key in pred_stage1.keys():
        if key not in pred_stage2:
            print('mismatch in x-y loc ' + fol)
            return

        x, y = key.split('_')
        x, y = float(x), float(y)
        x, y = int(x), int(y)

        c0_prob, c1_prob, c2_prob, c3_prob = pred_stage1[key][0], pred_stage1[key][1], 0, 0
        if pred_stage2[key][0] >= 0.5:
            c3_prob = pred_stage1[key][2]
            c2_prob = (1- pred_stage2[key][0])/pred_stage2[key][0]*c3_prob
        elif pred_stage2[key][0] > 0:
            c2_prob = pred_stage1[key][2]
            c3_prob = pred_stage2[key][0]/(1 - pred_stage2[key][0])*c2_prob

        fid.writelines('{} {} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(x, y, c0_prob, c1_prob, c2_prob, c3_prob))

    fid.close()


if __name__ == '__main__':
    out_fol = '../heatmap_txt'
    parent_fol = '../patches'
    if not is_path_exists(out_fol):
        os.mkdir(out_fol)

    source_fols = [os.path.join(parent_fol, fol) for fol in os.listdir(parent_fol) if '.tif' in fol]
    for fol in source_fols:
        print(fol)
        process_fol(fol, out_fol)
        slide_id = fol.split('/')[-1].split('.')[0]
        os.system('cp {} {}'.format(os.path.join(fol, 'patch-level-color.txt'),\
                        os.path.join(out_fol, 'color-' + slide_id)))

