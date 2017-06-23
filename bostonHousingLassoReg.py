import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import pandas as pd
from mglearn.datasets import load_extended_boston
# Lasso regression - (0 coefs implies automatics feature selection)
import numpy as np

X, y = load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train.shape)

lasso = Lasso().fit(X_train, y_train)

print("Training score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Features used: {:.2f}".format(np.sum(lasso.coef_ != 0)))

# Training score: 29%   Test Score: 21% Underfitting

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

print("Training score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Features used: {:.2f}".format(np.sum(lasso001.coef_ != 0)))
