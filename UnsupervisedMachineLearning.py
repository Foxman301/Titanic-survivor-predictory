# This program is meant to create an Unsupervised Kmeans Cluster with the
# data from the Titanic wreck, created by Jonathan Espedal
# Load the data
import seaborn as sns
titanic = sns.load_dataset('titanic')

# Drop the columns
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

#Remove the rows with missing values
titanic = titanic.dropna(subset =['embarked', 'age'])

#Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)

#Encode embarked
titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)

#Clustering Code
from itertools import cycle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy.random import RandomState
import pylab as pl

class clustering:
	def __init__(self):
		self.plot(titanic)

	def plot(self, X):
		pca = PCA(n_components=2, whiten=True).fit(X)
		X_pca = pca.transform(X)
		kmeans = KMeans(n_clusters=3, random_state=RandomState(42)).fit(X_pca)
		plot_2D(X_pca, kmeans.labels_, ["c0", "c1", "c2"])

def plot_2D(data, target, target_names):
	colors = cycle('rgbcmykw')
	target_ids = range(len(target_names))
	pl.figure()
	for i, c, label in zip(target_ids, colors, target_names):
		pl.scatter(data[target == i, 0], data[target == i, 1],
					c=c, label=label)
	pl.legend()
	pl.show()

if __name__ == '__main__':
	c = clustering()