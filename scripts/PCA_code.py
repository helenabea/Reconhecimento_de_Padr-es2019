import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from statsmodels import robust

X = pd.read_csv("data/X/normalized/GBM_X_normalized.tsv", delimiter='\t', index_col=0)
y = pd.read_csv("data/Y/Y_GBM_NF1.tsv", delimiter='\t')

# subset using MAD genes
mad = pd.Series.from_csv("tables/full_mad_genes.tsv", sep='\t')

# choose 8000 genes with the most varied expression
mad = mad[:8001]
X = X.loc[mad.index]
X = X.dropna()
X = X[y['Sample']]

# PCA
pca=PCA(n_components=2)
principalComponents = pca.fit_transform(X.T)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y['status']], axis = 1)

# make a plot and save
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Principal Component 1", fontsize = 15)
ax.set_ylabel("Principal Component 2", fontsize = 15)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['status'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
fig.savefig("figures/pca.png")
