from sklearn.linear_model import Ridge
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Experimenting with ridge regression - (near zero coefs)
# alpha val = def 1.0

X, y = load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train, y_train)

print("Training score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test score {:.2f}".format(ridge.score(X_test, y_test)))

# Ridge regression applies constraints to avoid overfitting - Training score: 89% Test score: 75%

# Using alpha = 10

ridge10 = Ridge(alpha=10).fit(X_train, y_train)

print("Training score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test score {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

print("Training score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test score {:.2f}".format(ridge01.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, '^', label= "Ridge alpha=0.1")
plt.plot(ridge10.coef_, 'v', label="Ridge alpha=10")

plt.xlabel("Coef index")
plt.ylabel("Coef magnitude")
plt.show()

# alpha = 0 = Linear Reg
