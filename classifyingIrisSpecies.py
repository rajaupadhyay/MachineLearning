import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
import scipy as sp
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Distinguishing species of Iris flowers (SUPERVISED LEARNING) (Classification)
# Features: Length/Width of Petals and Sepals
# Classes: setosa, versicolor, virginica

iris_dataset = load_iris()
# print(iris_dataset.keys())
# print(iris_dataset['DESCR'][:193])
# print(list(enumerate(iris_dataset['target_names'])))
# print(iris_dataset['data'][:5]) # sepal length, sepal width, petal length, petal width
# print(iris_dataset['target'][:5])

#Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print(X_train.shape)
print(X_test.shape)

# Visualising the data

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
# print(iris_dataframe)

grr = pd.scatter_matrix(iris_dataframe,c=y_train, figsize=(12,12),hist_kwds={'bins':20}, marker='x',s=60, alpha=0.8, cmap=mglearn.cm3)

plt.show()
