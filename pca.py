"""
Principal Component Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def create_dataset():
	x1 = np.linspace(0, 10, 200)
	x2 = x1 * 0.5 + 1.2 + np.random.normal(0, 1, x1.shape) * np.sqrt(0.1 / (np.abs(x1 - 5)))
	
	X = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
	plt.scatter(x1, x2)
	plt.savefig('./dataset.png')
	
	return X, x1, x2

def pca(X, n_components=2):
	pca = PCA(n_components=n_components)
	pca.fit(X)
	
	return pca.components_, pca.explained_variance_ratio_

def plot_components(x1, x2, pca_components):
	plt.scatter(x1, x2)
	pca_comp_1 = np.linspace(0, 12, 100)[:, np.newaxis] * pca_components[0, :]
	pca_comp_2 = np.linspace(-2, 2, 100)[:, np.newaxis] * pca_components[1, :]
	plt.plot(pca_comp_1[:, 0], pca_comp_1[:, 1], linewidth=2, c='r')
	plt.plot(pca_comp_2[:, 0], pca_comp_2[:, 1], linewidth=2, c='g')
	plt.savefig('./pca_components.png')

if __name__ == '__main__':
	X, x1, x2 = create_dataset()
	pca_components, explained_variance_ratio = pca(X)
	plot_components(x1, x2, pca_components)
	print('Number of components: \n%s \nand explained variance ratio: \n%s'%(pca_components, explained_variance_ratio))
