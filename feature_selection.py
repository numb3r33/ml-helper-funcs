"""
Feature Selection in case of anonymized features.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import f_classif
from sklearn.svm import l1_min_c
from sklearn.ensemble import RandomForestClassifier

def create_dataset():
	X, y = make_blobs(n_samples=5000, n_features=100, centers=2, cluster_std=10, random_state=241)
	
	return X, y

def get_low_variance_features_index(X):
	low_var_idx = np.std(X, axis=0).argsort()
	return low_var_idx

def get_low_variance_feature_values(X, low_var_idx, n=20):
	return np.std(X, axis=0)[low_var_idx][:n]

def plot_histogram(X, low_var_idx, index):
	plt.hist(X[:, low_var_idx[index]], bins=100)
	plt.xlabel('Feature with low variance')
	plt.show()

def count_non_zero_variance_features(X, low_var_idx, index):
	return np.sum(X[:, low_var_idx[index]] > 0)

def split_examples(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=214)
	
	return X_train, X_test, y_train, y_test

def measure_classification_quality(X, y, low_var_idx):
	X_train, X_test, y_train, y_test = split_examples(X, y)
	quality = []

	for i in range(0, 100, 4):
		clf = LogisticRegression(C=0.1)
		clf.fit(X_train[:, low_var_idx[i:]], y_train)
		q = roc_auc_score(y_test, clf.predict_proba(X_test[:, low_var_idx[i:]])[:, 1])
		quality.append((i, q))
	
	return quality

def feature_selection(X, y):
	var_imp = f_classif(X, y)[1]
	var_imp[np.isnan(var_imp)] = 1
	imp_feature_idx = var_imp.argsort()[::-1]
	
	print('Important feature indices: %s'%(imp_feature_idx))	

	return var_imp

def plot_feature_importance(var_imp):
	plt.plot(sorted(var_imp))
	plt.show()

def get_C_grid(X, y):
	c_grid = l1_min_c(X, y, loss='log') * np.logspace(0, 3, 100)
	return c_grid

def prediction_quality(X, y):
	X_train, X_test, y_train, y_test = split_examples(X, y)
	n_features = []
	quality = []
	
	c_grid = get_C_grid(X_train, y_train)

	for c in c_grid:
		clf = LogisticRegression(penalty='l1', C=c)
		clf.fit(X_train, y_train)
		q = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
		quality.append(q)
		n_features.append(np.sum(clf.coef_ > 0))
	
	return quality, n_features

def lasso_regression(X, y):
	"""
	Use Randomized Logistic Regression to select the features based on the coefficient values
	"""

	clf = RandomizedLogisticRegression(C=1.0)
	clf.fit(X, y)
	print('Number of non zero valued coefficients: ', np.sum(clf.scores_ > 0))
	imp_feature_idx = clf.scores_.argsort()
	
	qualities = []
	
	X_train, X_test, y_train, y_test = split_examples(X, y)
	
	for i in range(0, 100, 4):
		clf = LogisticRegression(C=0.1)
		clf.fit(X_train[:, imp_feature_idx[i:]], y_train)
		q = roc_auc_score(y_test, clf.predict_proba(X_test[:, imp_feature_idx[i:]])[:, 1])
		
		qualities.append(q)
	plt.plot(range(0, 100, 4), qualities)
	plt.show()
	
	return qualities

def ensemble_learning(X, y):
	"""
	Fit Random Forest Classifier and use variable importance
	"""
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X, y)
	
	plt.bar(range(100), clf.feature_importances_)
	plt.show()
	
	return clf.feature_importances_

if __name__ == '__main__':
	X, y = create_dataset()
	low_var_idx = get_low_variance_features_index(X)
	low_variance_features = get_low_variance_feature_values(X, low_var_idx)
	
	plot_histogram(X, low_var_idx, index=2)
	num_non_zero_variance_features = count_non_zero_variance_features(X, low_var_idx, index=2)
	
	print('Number of non zero values for feature with index = %d having low variance is %d'%(2, num_non_zero_variance_features))
	quality = measure_classification_quality(X, y, low_var_idx)
	print('Classification quality: \n%s'%(quality))
	
	imp_feature_idx = feature_selection(X, y)
	plot_feature_importance(imp_feature_idx)
	
	quality, n_features = prediction_quality(X, y)
	
	plt.plot(n_features, quality)	
	plt.xlabel('Num features with non zero coefficient values')
	plt.ylabel('Quality of the prediction')
	plt.show()
	
	# Lasso Regression for feature selection
	qualities = lasso_regression(X, y)
	
	# ensemble learning
	feature_importances_ = ensemble_learning(X, y)

