"""
Feature Selection in case of anonymized features.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
	plt.show()
	plt.savefig('./feature_histogram.png')

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

if __name__ == '__main__':
	X, y = create_dataset()
	low_var_idx = get_low_variance_features_index(X)
	low_variance_features = get_low_variance_feature_values(X, low_var_idx)
	
	plot_histogram(X, low_var_idx, index=2)
	num_non_zero_variance_features = count_non_zero_variance_features(X, low_var_idx, index=2)
	
	print('Number of non zero values for feature with index = %d having low variance is %d'%(2, num_non_zero_variance_features))
	quality = measure_classification_quality(X, y, low_var_idx)
	print('Classification quality: \n%s'%(quality)) 
