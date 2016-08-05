"""
How to define and pass custom metrics to a K-Neighbors
Classifier
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def euclidean_distance(a, b):
	return np.sum(np.power(a - b, 2.0))

def create_examples():
	X, y = make_blobs(n_samples=1000, n_features=20, centers=2, cluster_std=10, random_state=241)
	
	return X, y

def split_dataset(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=241)
	
	return X_train, X_test, y_train, y_test

def train_knn():
	X, y = create_examples()
	X_train, X_test, y_train, y_test = split_dataset(X, y)
	
	# pass in the custom metric in the metric parameter
	clf = KNeighborsClassifier(n_neighbors=5, metric=euclidean_distance)
	
	# train the classifier on the dataset
	clf.fit(X_train, y_train)
	print('Accuracy Score: %f'%(accuracy_score(y_test, clf.predict(X_test))))

if __name__ == '__main__':
	train_knn()
