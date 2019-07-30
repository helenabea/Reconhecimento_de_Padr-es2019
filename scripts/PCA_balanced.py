import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from statsmodels import robust

X = pd.read_csv("data/X/normalized/GBM_X_normalized.tsv", delimiter='\t', index_col=0)
y = pd.read_csv("data/Y/Y_GBM_NF1.tsv", delimiter='\t', index_col=1)

# subset using MAD genes
mad = pd.Series.from_csv("tables/full_mad_genes.tsv", sep='\t')
# choose 8000 genes with the most varied expression
mad = mad[:8001]
X = X.loc[mad.index]
X = X.dropna()
X = X[y.index]

# Join X and Y
df = pd.concat([X.T, y['status']], axis = 1)

count_class_0, count_class_1 = df.status.value_counts()

df_class_0 = df[df['status'] == 0]
df_class_1 = df[df['status'] == 1]

# Random under-smapling
df_class_0_under = df_class_0.sample(count_class_1)
df_under = pd.concat([df_class_0_under, df_class_1], axis=0)

#df_under.status.value_counts().plot(kind='bar', title='Count (status)')

# Separate input features (X) and target variable (y)
dfY = df_under.status
dfX = df_under.drop('status', axis=1)

# PCA
pca=PCA(n_components=2)
principalComponents = pca.fit_transform(dfX)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.index = dfX.index
finalDf = pd.concat([principalDf, dfY], axis = 1)

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
fig.savefig("figures/pca_under.png")

# Random over-sampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_over = pd.concat([df_class_0, df_class_1_over], axis=0)

# Separate input features (X) and target variable (y)
dfY = df_over.status
dfX = df_over.drop('status', axis=1)

# PCA
pca=PCA(n_components=2)
principalComponents = pca.fit_transform(dfX)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.index = dfX.index
finalDf = pd.concat([principalDf, dfY], axis = 1)

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
fig.savefig("figures/pca_over.png")
