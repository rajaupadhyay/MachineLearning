from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import mglearn
import numpy as np

X, y = make_blobs(random_state=42)

mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()

linear_svc = LinearSVC().fit(X, y)
print("Coeff shape: ", linear_svc.coef_.shape)
print("Intercept shape: ", linear_svc.intercept_.shape)

mglearn.discrete_scatter(X[:,0], X[:,1],y)
line = np.linspace(-15, 15)

for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_, ['b','r','g']):
    plt.plot(line, -(line*coef[0]+intercept)/coef[1],c=color)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Line Class 0", "Line Class 1", "Line Class 2"])
plt.show()