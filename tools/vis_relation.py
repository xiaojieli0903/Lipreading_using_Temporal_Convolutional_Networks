import numpy as np
import os
import sys
import pandas as pd
import seaborn
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

input_dir = sys.argv[1]
output_dir = sys.argv[2]

features = sorted(os.listdir(input_dir))
# random.shuffle(features)
fig = plt.figure()

for idx in range(10):
    item = features[idx]
    feature = np.load(os.path.join(input_dir, item))['data']
    feature = feature.reshape(-1, feature.shape[-1])
    seq_len = feature.shape[0]
    correlation = np.corrcoef(feature) #cosine_similarity(feature)
    mask = np.ones_like(correlation)
    mask[np.triu_indices_from(mask)] = False

    seaborn.heatmap(correlation, mask=mask, center=0, annot=False, cmap='YlGnBu')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/{idx}_{item}.jpg', dpi=200)
    plt.figure().clear()
    print(seq_len, feature.shape)
