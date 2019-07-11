import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams.update({'font.size': 12})

#x = np.load('thresholds_dice_saved.npy')

source = 'v1_results'
#source = '.'

files = [f for f in os.listdir(source) if ('.npy' in f) and ('_4' in f or 'high' in f or 'HASHI' in f)]

leds = {'high_res':'Original Heatmap', 'low_res_max_2':'Aggregated_Max_2', 'low_res_max_4':'Aggregated_Max_4', 'low_res_max_8':'Aggregated_Max_8',
        'low_res_avg_2':'Aggregated_Average_2', 'low_res_avg_4':'Aggregated_Average_4', 'low_res_avg_8':'Aggregated_Average_8',
        'low_res_median_2':'Aggregated_Median_2', 'low_res_median_4':'Aggregated_Median_4', 'low_res_median_8':'Aggregated_Median 8',
        'smooth7':'Ours', 'HASHI_seer':'HASHI_seer', 'HASHI_seer_aggregated':'HASHI_seer_aggregated'
        }

files = ['smooth7_thresholds_dice_saved.npy',
         'HASHI_seer_aggregated_thresholds_dice_saved.npy',
         'HASHI_seer_thresholds_dice_saved.npy']

legends = []
y = np.loadtxt('dice_sota.txt')
plt.plot(y[:, 0], y[:, 1], '-^', linewidth=1, markevery=10, markersize=2)
legends.append('HASHI')

styles = ['-*', '-o', '-^', '-.', '--']

for ind, fn in enumerate(files):
    x = np.load(os.path.join(source, fn))
    print('x: ', x[0, :])
    dum = np.array([1, 0]).reshape(1,2)
    x = np.vstack((x, dum))
    plt.plot(x[:,0], x[:,1], styles[ind], linewidth=1, markevery=10, markersize=5)
    print(fn.split('_thresholds')[0])
    legends.append(leds[fn.split('_thresholds')[0]])

plt.xlim([0.01, 1])
plt.ylim([0.01, 1])
plt.xlabel('Threshold')
plt.ylabel('Avg. Dice coefficient')
plt.xticks(np.arange(0, 1, step=0.1))
plt.yticks(np.arange(0, 1, step=0.1))
plt.legend(legends, loc=3)
plt.grid(linestyle='--')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


