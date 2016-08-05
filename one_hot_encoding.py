"""
One Hot Encoding is the common approach taken when working with categorical variables.
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def create_dataset():
	"""
	Creates a toy dataset
	"""
	simple_data = pd.DataFrame({'City': ['Msk', 'Spb', 'Msk'],\
								'Weather': ['good', 'bad', 'worse'] })	
	
	return simple_data

def one_hot_encode(df):
	"""
	Takes in a data frame containing categorical features
	and converts them into one hot encoded features

	Parameter :
		df: DataFrame containing categorical features
	
	Return :
		One hot encoded representation of the categorical
		features.
	
	"""
	
	# transform every object in the dict
	simple_data_dict = df.T.to_dict().values()
	
	transformer = DictVectorizer(sparse=False)
	# Apply this transformer to the data to get a binary array
	
	return transformer.fit_transform(simple_data_dict)

if __name__ == '__main__':
	data = create_dataset()
	print('Dataset: \n', data)
	
	one_hot_encoded = one_hot_encode(data)
	print('One Hot Encoded: \n', one_hot_encoded)
