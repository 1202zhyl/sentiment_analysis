# -*- coding： utf-8

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

wordMatrix = np.load('vocabularyMatrix.npy')
vocabulary = np.load('vocabulary.npy').tolist()
vocabulary = [(w.encode('UTF-8')).decode('UTF-8') for w in vocabulary]

labels = []
tokens = []

for i in range(1000, 1300):
    tokens.append(wordMatrix[i])
    labels.append(vocabulary[i])

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16))
for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()
