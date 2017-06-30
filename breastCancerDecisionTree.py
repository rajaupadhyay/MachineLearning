from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz
import graphviz
# Breast cancer classification using decision trees

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)

print("Training Accuracy {:.2f}".format(tree.score(X_train, y_train)))
print("Testing Accuracy {:.2f}".format(tree.score(X_test, y_test)))
# Pure leaves results in 100% Training accuracy - 94% Testing accuracy


# Setting max depth reduces over fitting (Pre-pruning)
tree1 = DecisionTreeClassifier(max_depth=4, random_state=0)

tree1.fit(X_train, y_train)

print("Training Accuracy {:.2f}".format(tree1.score(X_train, y_train)))
print("Testing Accuracy {:.2f}".format(tree1.score(X_test, y_test)))

export_graphviz(tree1, out_file="tree.dot", class_names=["malignant","benign"], feature_names=cancer.feature_names, impurity=False, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

print("Feature importances {}".format(tree1.feature_importances_))

