import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
import scipy as sp
import sklearn as sk
from sklearn.datasets import load_iris

# Distinguishing species of Iris flowers (SUPERVISED LEARNING) (Classification)
# Features: Length/Width of Petals and Sepals
# Classes: setosa, versicolor, virginica

iris_dataset = load_iris()
print(iris_dataset.keys())
print(iris_dataset['DESCR'][:193])
print(list(enumerate(iris_dataset['target_names'])))
print(iris_dataset['data'][:5]) # sepal length, sepal width, petal length, petal width
print(iris_dataset['target'][:5])

