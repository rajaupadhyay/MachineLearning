# Studying the breast cancer dataset using Multi layer perceptron (Neural Network)
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Need to test with keras, lasgana, tensor-flow (*)

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)

mlp.fit(X_train, y_train)

print("Accuracy on training set:", mlp.score(X_train, y_train))
print("Accuracy on test set:", mlp.score(X_test, y_test))

# Neural Networks expect data to have µ of 0 and var of 1

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train-mean_on_train)/std_on_train

X_test_scaled = (X_test-mean_on_train)/std_on_train

mlp1 = MLPClassifier(max_iter=1000,random_state=0)
mlp1.fit(X_train_scaled, y_train)

print("Accuracy on training set:", mlp1.score(X_train_scaled, y_train))
print("Accuracy on test set:", mlp1.score(X_test_scaled, y_test))

