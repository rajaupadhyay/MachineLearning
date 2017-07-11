from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100,noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train,fill = True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

mlp1 = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mlp1.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp1, X_train,fill = True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

mlp2 = MLPClassifier(solver='lbfgs',activation='tanh',random_state=0,hidden_layer_sizes=[10, 10])
mlp2.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp2, X_train,fill = True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()