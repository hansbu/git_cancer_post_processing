#!bin/bash

python -u gen_pred_gt_heatmap.py > log.gen_gt.txt
wait

python -u dice_auc_cal.py > log.compute_dice.txt
