import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as Lines

plt.rcParams.update({'font.size': 12})
hashi = [f.strip() for f in open('v1_results/HASHI_max4_dice_slide_level.txt', 'r')]
print(hashi)

ours = [f.strip() for f in open('v1_results/ours_max4_dice_slide_level.txt', 'r')]

hashi_h = {}; ours_h = {}
for i in range(len(ours)):
    hashi_h[hashi[i].split(',')[0]] = float(hashi[i].split(',')[1])
    ours_h[ours[i].split(',')[0]] = float(ours[i].split(',')[1])

dices = np.zeros((len(ours), 2))
slides = []
for i in range(len(ours)):
    slide = hashi[i].split(',')[0]
    slides.append(slide)
    dices[i, :] = np.array([hashi_h[slide], ours_h[slide]])

subs = abs(dices[:,0] - dices[:,1])
ind_max = np.argmax(subs)
print('hashi/ours: ', slides[ind_max], hashi_h[slides[ind_max]],  ours_h[slides[ind_max]])


fig, ax = plt.subplots()

plt.scatter(dices[:,0], dices[:,1])
plt.xlabel('HASHI')
plt.ylabel('ours')
plt.title('Agreement between HASHI and our algorithm')



line1 = [(0,0), (1,1)]

# Note that the Line2D takes a list of x values and a list of y values,
# not 2 points as one might expect.  So we have to convert our points
# an x-list and a y-list.
(line1_xs, line1_ys) = zip(*line1)

ax.add_line(Lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
plt.plot()

plt.show()