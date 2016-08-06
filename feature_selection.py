"""
Feature Selection in case of anonymized features.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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


if __name__ == '__main__':
	X, y = create_dataset()
	low_var_idx = get_low_variance_features_index(X)
	low_variance_features = get_low_variance_feature_values(X, low_var_idx)
	
	plot_histogram(X, low_var_idx, index=2)
	num_non_zero_variance_features = count_non_zero_variance_features(X, low_var_idx, index=2)
	
	print('Number of non zero values for feature with index = %d having low variance is %d'%(2, num_non_zero_variance_features)) 
