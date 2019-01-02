from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
#[Height, Weight, shoe size]

X = [[181,80,44],[161,60,43],[191,80,42],[131,86,41],[181,88,44],[181,86,44],[171,84,44],[161,82,44],[141,81,44],[141,81,44]]

Y = ['male','female','female','female','male','male','male','male','female','male']

# Initialization of classifier
clf = tree.DecisionTreeClassifier()
clf_svc = SVC()
clf_KNN = KNeighborsClassifier()
clf_perception = Perceptron()

#Train model for the given hypothesis
clf = clf.fit(X,Y)
clf_svc = clf_svc.fit(X,Y)
clf_KNN = clf_KNN.fit(X,Y)
clf_perception = clf_perception.fit(X,Y)

#Test the data using DecisionTreeClassifier
pred_tree = clf.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy Decision Tree Classifier: {}'.format(acc_tree))

#Test the data using SVC
pred_svc = clf_svc.predict(X)
acc_svc = accuracy_score(Y, pred_svc) * 100
print('Accuracy SVC: {}'.format(acc_svc))

#Test the data using DecisionTreeClassifier
pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy KNeighbors Classifier: {}'.format(acc_KNN))

#Test the data using DecisionTreeClassifier
pred_perception = clf_perception.predict(X)
acc_perception = accuracy_score(Y, pred_perception) * 100
print('Accuracy Perceptron: {}'.format(acc_perception))

#prediction = clf.predict([[181,88,44]])
#print prediction

# The best classifier from svm, per, KNN, DecisionTree
index = np.argmax([acc_svc, acc_perception, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best classifier for Gender is {}'.format(classifiers[index]))