import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import scipy as sp
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Distinguishing species of Iris flowers (SUPERVISED LEARNING) (Classification)
# Features: Length/Width of Petals and Sepals
# Classes: setosa, versicolor, virginica

iris_dataset = load_iris()
# print(iris_dataset.keys())
# print(iris_dataset['DESCR'][:193])
print(list(enumerate(iris_dataset['target_names'])))
# print(iris_dataset['data'][:5]) # sepal length, sepal width, petal length, petal width
# print(iris_dataset['target'][:5])

#Splitting the data for training(75%) and testing(25%)
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print(X_test)
# print(y_test)

# Visualising the data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
# print(iris_dataframe)

grr = pd.scatter_matrix(iris_dataframe,c=y_train, figsize=(12,12),hist_kwds={'bins':20}, marker='x',s=60, alpha=0.8, cmap=mglearn.cm3)

# plt.show()

# k-nearest neighbors - k=1 for now

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(knn)

X_new = np.array([[5, 2.9, 1, 0.2]])

# Predicting the species with "unseen" data
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Species name: {}".format(iris_dataset['target_names'][prediction]))

# Evaluating the model using the test data
y_pred = knn.predict(X_test)
print("Test set preds: {}".format(y_pred))

print("Classifier Accuracy/Score: {:.2f}".format(knn.score(X_test, y_test)))
