import os
import sys

import matplotlib.pyplot as plt
import numpy as np

input_dir = sys.argv[1]
keywords = sys.argv[2]

f = open(input_dir, 'r')
lines = f.readlines()
losses = {}
for line in lines:
    if line.find(keywords) >= 0:
        # loss_cosine: 1.0000 * 0.2694
        if line.find('acc') < 0:
            loss = float(line.split('* ')[-1].strip())
            loss_name = line.split('-----')[-1].split(':')[0].strip()
        else:
            loss = float(line.split('* ')[-1].split(',')[0])
            loss_name = line.split('-----')[1].split(',')[0].split(':')[0]
        if loss_name not in losses:
            losses[loss_name] = [loss]
        else:
            losses[loss_name].append(loss)

fig_all, ax_all = plt.subplots(figsize=(12, 7))

for loss_name in losses:
    fig, ax = plt.subplots(figsize=(12, 7))
    value_number = len(losses[loss_name])
    x = np.arange(0, value_number, 1)
    ax.plot(x, losses[loss_name], linewidth=1, label=loss_name)
    ax_all.plot(x, losses[loss_name], linewidth=1, label=loss_name)
    ax.set_xlabel('iter')
    ax.legend(loc="best")
    fig.savefig(f'{os.path.dirname(input_dir)}/{loss_name}.jpg', dpi=200)
    fig.clear()

ax_all.set_xlabel('iter')
ax_all.legend(loc="best")
fig_all.savefig(f'{os.path.dirname(input_dir)}/all.jpg')
