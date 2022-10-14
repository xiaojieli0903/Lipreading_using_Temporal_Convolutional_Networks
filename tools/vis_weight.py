import torch
import os

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn
model_path = sys.argv[1]
key = sys.argv[2]

model = torch.load(model_path)['model_state_dict']
param = model[key]
print(param.shape)
feature_norms = torch.norm(param, dim=1)
correlation = np.corrcoef(param.cpu().numpy())
mask = np.eye(correlation.shape[0])

correlation = -(correlation - mask)
seaborn.heatmap(correlation, mask=None, center=0, annot=False, cmap='YlGnBu')
plt.savefig(f'tools/weight_vis.jpg', dpi=200)

print(f'feature norms ({feature_norms.shape}): {feature_norms}')
