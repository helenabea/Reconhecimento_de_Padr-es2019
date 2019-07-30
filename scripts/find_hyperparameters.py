import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import interp

def find_hyper_params_neural_net(X, y):

	solver = ['lbfgs', 'sgd', 'adam']
	alpha = [1e-5, 1e-4, 1e-3, 1e-2]
	hidden_layer_sizes = [(2, 2), (2, 5), (5, 2), (5, 5),
						(2, 2, 2), (2, 5, 2), (2, 2, 5),
						(2, 5, 5), (5, 2, 5),
						(5, 5, 5)]

	save = []

	X_train, X_test, y_train, y_test = train_test_split(X.T, y['status'],
			test_size = 0.33, random_state = 42)

	for s in solver:
		for a in alpha:
			for hs in hidden_layer_sizes:
				# Neural Network
				nn = MLPClassifier(solver = s, alpha = a,
					hidden_layer_sizes = hs, random_state = 1, max_iter = 1000)
				# calculate area under curve
				probas_test_ = nn.fit(X_train.values, y_train.values).predict_proba(X_test.values)
				fpr, tpr, thresholds = roc_curve(y_test.values, probas_test_[:, 1])
				roc_auc_test = auc(fpr, tpr)
				tmp = []
				tmp.append(roc_auc_test)
				tmp.append(s)
				tmp.append(a)
				tmp.append(hs)
				save.append(tmp)

	return save

def find_hyper_params_svm(X, y):

	kernel = ['rbf', 'sigmoid']
	gamma = ['scale', 'auto']

	save = []

	X_train, X_test, y_train, y_test = train_test_split(X.T, y['status'],
			test_size = 0.33, random_state = 42)

	for k in kernel:
		for g in gamma:
			# Neural Network
			svm = SVC(kernel = k, gamma = g, probability=True)
			# calculate area under curve
			probas_test_ = svm.fit(X_train.values, y_train.values).predict_proba(X_test.values)
			fpr, tpr, thresholds = roc_curve(y_test.values, probas_test_[:, 1])
			roc_auc_test = auc(fpr, tpr)
			tmp = []
			tmp.append(roc_auc_test)
			tmp.append(k)
			tmp.append(g)
			save.append(tmp)

	return save
