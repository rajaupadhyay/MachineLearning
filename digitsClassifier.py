import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import mglearn
import scipy as sp
import sklearn as sk
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits_dataset = load_digits()

# print(digits_dataset['DESCR'])
# print(digits_dataset['target_names'])
# print(digits_dataset['data'][0])

n_samples = len(digits_dataset.images)
data = digits_dataset.images.reshape((n_samples, -1))

imgplot = plt.imshow(digits_dataset['images'][20])
testVal = np.array(digits_dataset['data'][20])
testVal = testVal.reshape(1,-1)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits_dataset['data'], digits_dataset['target'], random_state=0)
# print(X_test[0])


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print("Test set preds: {}".format(y_pred))
print("Classifier Accuracy/Score: {:.2f}".format(knn.score(X_test, y_test)))

print(digits_dataset['target'][20])
z_pred = knn.predict(testVal)
print("Data point prediction: {}".format(z_pred))