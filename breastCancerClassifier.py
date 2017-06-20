import sklearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
print(cancer['DESCR'])
print(cancer['data'][0])
print(cancer['target_names'])

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,stratify=cancer.target,random_state=66)

training_accuracy = []
test_accuracy = []
neighbors = range(1,11)

for n in neighbors:
    knnClassifier = KNeighborsClassifier(n_neighbors=n)
    knnClassifier.fit(X_train,y_train)
    training_accuracy.append(knnClassifier.score(X_train,y_train))
    test_accuracy.append(knnClassifier.score(X_test,y_test))

plt.plot(neighbors,training_accuracy,label="training accuracy")
plt.plot(neighbors,test_accuracy,label="testing accuracy")
plt.xlabel("Neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
