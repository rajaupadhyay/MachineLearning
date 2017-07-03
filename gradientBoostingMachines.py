from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
# Gradient boosted decision trees on breast cancer dataset
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,random_state=0)

grbt = GradientBoostingClassifier(random_state=0)

grbt.fit(X_train, y_train)

print("Training set accuracy: {:.2f}".format(grbt.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(grbt.score(X_test, y_test)))
# 100% accuracy on training set: Overfitting - We can lower the learning rate to reduce overfitting or use stronger
# pre-pruning (limit max depth)

grbt1 = GradientBoostingClassifier(random_state=0, max_depth=1)
grbt1.fit(X_train, y_train)

print("Training set accuracy: {:.2f}".format(grbt1.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(grbt1.score(X_test, y_test)))

grbt2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
grbt2.fit(X_train, y_train)

print("Training set accuracy: {:.2f}".format(grbt2.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(grbt2.score(X_test, y_test)))

