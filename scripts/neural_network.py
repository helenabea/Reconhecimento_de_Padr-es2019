import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import interp

# support script in the same folder
from find_hyperparameters import find_hyper_params_neural_net

#############
# Functions #
#############
def roc(probas_, X, Y, aucs, tprs):
    fpr, tpr, thresholds = roc_curve(Y, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    return fpr,tpr

def mean_auroc(tprs, aucs):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    return mean_tpr, mean_auc, std_auc, tprs_upper,tprs_lower

#################
# Start of Code #
#################

# read files
X = pd.read_csv("data/X/normalized/GBM_X_normalized.tsv", delimiter='\t', index_col=0)
y = pd.read_csv("data/Y/Y_GBM_NF1.tsv", delimiter='\t')

# subset using MAD genes
mad = pd.Series.from_csv("tables/full_mad_genes.tsv", sep='\t')

# choose 8000 genes with the most varied expression
mad = mad[:8001]
X = X.loc[mad.index]
X = X.dropna()
# get sample with class
X = X[y['Sample']]

###############################
# Select best hyperparameters #
###############################
options = find_hyper_params_neural_net(X, y)
# make a dataframe
df_options = pd.DataFrame(options)
# select best hyperparameters
s = df_options.iloc[df_options[0].idxmax()][1]
a = df_options.iloc[df_options[0].idxmax()][2]
hs = df_options.iloc[df_options[0].idxmax()][3]

print("Rede Neural, Solver=", s, ", alpha=", a, ", hidden layer sizes=", hs)

# prepare for ROC
tprs_test = []
aucs_test = []
tprs_train = []
aucs_train = []
mean_fpr = np.linspace(0, 1, 100)

# Neural Network
# with best hyperparameters
nn = MLPClassifier(solver = s, alpha = a, hidden_layer_sizes = hs,
                    random_state=1, max_iter = 1000)

# 500 different classifiers
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=1)

for train_index, test_index in rskf.split(X.T, y['status']):
    # get indexes
    X_train, X_test = X.T.values[train_index], X.T.values[test_index]
    # get values
    y_train, y_test = y['status'].values[train_index], y['status'].values[test_index]
    # fit and test
    probas_ = nn.fit(X_train, y_train).predict_proba(X_test)
    # plot ROC for test dataset
    fpr_test, tpr_test = roc(probas_, X_test, y_test, aucs_test, tprs_test)
    plt.plot(fpr_test, tpr_test, lw=0.3, alpha=0.01,
            label=None, color='#F8766D')
    # fit and test for train
    probas_ = nn.fit(X_train, y_train).predict_proba(X_train)
    # fit and plot ROC for train dataset
    fpr_train, tpr_train = roc(probas_, X_train, y_train, aucs_train, tprs_train)
    plt.plot(fpr_train, tpr_train, lw=0.3, alpha=0.01,
            label=None, color='#00BFC4')

# plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance', alpha=.8)

# Mean AUROC for test
mean_tpr, mean_auc, std_auc, tprs_upper,tprs_lower = mean_auroc(tprs_test, aucs_test)
plt.plot(mean_fpr, mean_tpr, color='#F8766D',
         label=r'Mean ROC Test (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
# Standard Deviation for test
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

# Mean AUROC for train
mean_tpr, mean_auc, std_auc, tprs_upper,tprs_lower = mean_auroc(tprs_train, aucs_train)
plt.plot(mean_fpr, mean_tpr, color='#00BFC4',
         label=r'Mean ROC Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
# Standard Deviation for train
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=None)

# Final plot
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.savefig("figures/roc_neural_net.png")

##############
# Validation #
##############

#open tcga micro-array file
X = pd.read_csv("data/X/tdm/GBM.tsv", delimiter='\t', index_col=0)
X = X.loc[mad.index]
X = X.fillna(0)
# get sample with class
X = X[y['Sample']]

# open validation file
X_validation = pd.read_csv("data/validation/validation_set.tsv", delimiter='\t', index_col=0)
X_validation = X_validation.loc[mad.index]
X_validation = X_validation.fillna(0)

nn = MLPClassifier(solver = 'sgd', alpha = 0.0001, hidden_layer_sizes = (2, 5, 5),
                    random_state=1, max_iter = 1000)

nn.fit(X.T.values, y['status'].values)

# normalization of validation dataset
scaler = StandardScaler()
scaler.fit(X_validation.T.values)
X_validation_standard = scaler.transform(X_validation.T.values)

probas_ = nn.predict_proba(X_validation_standard)

df = pd.DataFrame(probas_, index = X_validation.T.index)

print(df)
