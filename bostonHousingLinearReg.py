from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mglearn import datasets
import pandas as pd
# Using the extended Boston housing dataset

X, y = datasets.load_extended_boston()
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)

print("Training Score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test score: {:.2f}".format(lr.score(X_test, y_test)))
# Overfitting - training score: 95% test score: 61%
