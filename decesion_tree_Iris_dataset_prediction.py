#Code taken from
#https://towardsdatascience.com/decision-tree-in-python-b433ae57fb93

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

X.head()

#to create a confusion matrix at a later point
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#create and train an instance of the DecisionTreeClassifer class
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


#view the actual decision tree produced by our model

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, feature_names=iris.feature_names)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#how our decision tree does when its presented with test data
y_pred = dt.predict(X_test)


#confusion matrix.
species = np.array(y_test).argmax(axis=1)
predictions = np.array(y_pred).argmax(axis=1)
confusion_matrix(species, predictions)

#our decision tree classifier correctly classified 37/38 plants.

print('Scores: %s' % predictions)

# Create PNG
graph.write_png("iris.png")