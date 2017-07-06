from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
# Using Radial basis function kernel instead of polynomial kernel

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("Training Accuracy {:.2f}".format(svc.score(X_train, y_train)))
print("Test accuracy {:.2f}".format(svc.score(X_test, y_test)))


# plt.plot(X_train.min(axis=0), 'o', label="min")
# plt.plot(X_train.max(axis=0), '^', label="max")
# plt.xlabel("Feature index")
# plt.ylabel("Feature magnitude")
# plt.yscale("log")
# plt.show()

# Rescaling the dataset

minTraining = X_train.min(axis=0)
rangeTraining = (X_train-minTraining).max(axis=0)

X_train_scaled = (X_train-minTraining)/rangeTraining

X_test_scaled = (X_test-minTraining)/rangeTraining

svc1 = SVC()
svc1.fit(X_train_scaled, y_train)

print("Training Accuracy {:.2f}".format(svc1.score(X_train_scaled, y_train)))
print("Test accuracy {:.2f}".format(svc1.score(X_test_scaled, y_test)))
