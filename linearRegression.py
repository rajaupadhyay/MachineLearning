from sklearn.linear_model import LinearRegression
from mglearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X, y = datasets.make_wave(n_samples=60)
df = pd.DataFrame(X,columns=["feature1"])
df.insert(1,"output",y)
print(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

# check weights and y-axis offset

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept: {}".format(lr.intercept_))

sns.lmplot(x="feature1", y="output", data=df, size=4, scatter_kws={"s": 50, "alpha": 1})
plt.show()

print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# Underfitting - 67% training score and 66% testing score

