import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score


def confusion_matrix(Or, Tr, thres):
    tpos = np.sum((Or>=thres) * (Tr==1))
    tneg = np.sum((Or< thres) * (Tr==0))
    fpos = np.sum((Or>=thres) * (Tr==0))
    fneg = np.sum((Or< thres) * (Tr==1))
    return tpos, tneg, fpos, fneg

def auc_roc(Pr, Tr):
    fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0)
    return auc(fpr, tpr)

def dice_score(Pr, Tr, thres):
    im1 = Pr.copy()
    im2 = Tr.copy()

    im1[im1 > thres] = 1
    im1[Pr <= thres] = 0
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / (im1.sum() + im2.sum())

def cut_off(x):
    if x <= 0:
        print('Cut_off is applied')
        return float('inf')
    return x


def compute_dice(out_fol, steps=100):
    files = [x for x in os.listdir(out_fol) if 'npy' in x]
    # steps = 100
    thresholds = np.array(range(30, 60))/steps
    # thresholds = np.array([0.4, 0.5, 0.6])

    dice_thresholds = np.zeros(len(thresholds))


    print(thresholds)

    for i, threshold in enumerate(thresholds):        # compute dice by varying thresholds
        dice_running = 0
        for file in files:
            data = np.load(os.path.join(out_fol, file))
            im1 = data[:, 0:1]
            im2 = data[:, 1:]
            dice_running += dice_score(im1, im2, threshold)

        dice_thresholds[i] = dice_running/len(files)
        print('applying threshold: ', threshold, dice_running/len(files))
        sys.stdout.flush()

    print(dice_thresholds)
    save_dice = np.concatenate((thresholds.reshape(-1, 1), dice_thresholds.reshape(-1, 1)), axis = 1)
    np.save(os.path.join(out_fol, out_fol.split('/')[-1] + '_thresholds_dice_saved'), save_dice)

    threshold_best = thresholds[np.argmax(dice_thresholds)]

    stats = np.zeros((len(files), 7))
    for i, file in enumerate(files):
        data = np.load(os.path.join(out_fol, file))
        im1 = data[:, 0]
        im2 = data[:, 1]

        dice = dice_score(im1.copy(), im2.copy(), threshold_best)
        tpos, tneg, fpos, fneg = confusion_matrix(im1.copy(), im2.copy(), threshold_best)
        PPV = tpos / cut_off(tpos + fpos)
        NPV = tneg / cut_off(fneg + tneg)
        TPR = tpos / cut_off(tpos + fneg)
        FNR = fneg / cut_off(tpos + fneg)
        FPR = fpos / cut_off(fpos + tneg)
        TNR = tneg / cut_off(fpos + tneg)
        stats[i] = np.array([dice, PPV, NPV, TPR, TNR, FPR, FNR])

        # auc_val = auc_roc(im1.copy(), im2.copy())
        # print(file, auc_val, dice, '\n')

    stats = stats[:i+1]
    mu = np.mean(stats, axis = 0)
    stds = np.std(stats, axis = 0)

    with open(os.path.join(out_fol, out_fol.split('/')[-1] + '_results_summary.txt'), 'w') as f:
        f.writelines('best threshold: {}\n'.format(threshold_best))
        f.writelines('mean: {}\n'.format(mu))
        f.writelines('stds: {}\n'.format(stds))

    print('best threshold: ', threshold_best)
    print('mean: ', mu)
    print('stds: ', stds)


def compute_dice_final(out_fol, threshold_best):
    files = [x for x in os.listdir(out_fol) if 'preds_gts.npy' in x]

    stats = np.zeros((len(files), 7))
    for i, file in enumerate(files):
        data = np.load(os.path.join(out_fol, file))
        im1 = data[:, 0]
        im2 = data[:, 1]

        dice = dice_score(im1.copy(), im2.copy(), threshold_best)
        tpos, tneg, fpos, fneg = confusion_matrix(im1.copy(), im2.copy(), threshold_best)
        print(tpos, tneg, fpos, fneg)

        PPV = tpos / cut_off(tpos + fpos)
        NPV = tneg / cut_off(fneg + tneg)
        TPR = tpos / cut_off(tpos + fneg)
        FNR = fneg / cut_off(tpos + fneg)
        FPR = fpos / cut_off(fpos + tneg)
        TNR = tneg / cut_off(fpos + tneg)
        stats[i] = np.array([dice, PPV, NPV, TPR, TNR, FPR, FNR])

        # auc_val = auc_roc(im1.copy(), im2.copy())
        print(file, stats[i])

    stats = stats[:i+1]
    mu = np.mean(stats, axis = 0)
    stds = np.std(stats, axis = 0)

    with open(os.path.join(out_fol, out_fol.split('/')[-1] + '_slides_level.txt'), 'w') as f:
        for i, s in enumerate(stats):
            f.writelines('{},{},{},{},{},{},{},{}\n'.format(files[i].split('_preds')[0],s[0],s[1],s[2],s[3],s[4],s[5],s[6]))


    with open(os.path.join(out_fol, out_fol.split('/')[-1] + '_results_summary.txt'), 'w') as f:
        f.writelines('best threshold: {}\n'.format(threshold_best))
        f.writelines('mean: {}\n'.format(mu))
        f.writelines('stds: {}\n'.format(stds))

    print('best threshold: ', threshold_best)
    print('mean: ', mu)
    print('stds: ', stds)

