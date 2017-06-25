from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

logreg = LogisticRegression().fit(X_train, y_train)

print("Training score: {:.2f}".format(logreg.score(X_train, y_train)))
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))

# Training score: 96%  Test score: 96% (UNDERFITTING)

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)

print("Training score: {:.2f}".format(logreg100.score(X_train, y_train)))
print("Test score: {:.2f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)

print("Training score: {:.2f}".format(logreg001.score(X_train, y_train)))
print("Test score: {:.2f}".format(logreg001.score(X_test, y_test)))

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg001.coef_.T, 'x', label="C=0.01")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()

plt.show()