# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 07:41:49 2019

@author: SOMESH
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Classifying with hard voting

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr', log_clf),
                                          ('rf', rnd_clf), ('svc', svm_clf)],
                              voting='hard')
voting_clf.fit(X_train, y_train)

print('Classifying with hard voting\n')
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    



'''
Classifying with soft voting

If all classifiers are able to estimate class probabilities (i.e., they have a predict_proba()
 method), then you can tell Scikit-Learn to predict the class with the highest class probability,
 averaged over all the individual classifiers. This is called soft voting. It often achieves
 higher performance than hard voting because it gives more weight to highly confident votes. 
 All you need to do is replace voting="hard" with voting="soft" and ensure that all classifiers 
 can estimate class probabilities. This is not the case of the SVC class by default, so you need 
 to set its probability hyperparameter to True (this will make the SVC class use cross-validation
 to estimate class probabilities, slowing down training, and it will add a predict_proba() method).

'''
    
    
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(gamma="auto", probability=True, random_state=42)    
    
    
voting_clf = VotingClassifier(estimators=[('lr', log_clf),
                                          ('rf', rnd_clf), ('svc', svm_clf)],
                              voting='soft')
voting_clf.fit(X_train, y_train)
print('\nClassifying with soft voting\n')

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
'''
Bagging and Pasting

The following code trains an
ensemble of 500 Decision Tree classifiers,5 each trained on 100 training instances randomly
sampled from the training set with replacement (this is an example of bagging,
but if you want to use pasting instead, just set bootstrap=False). The n_jobs parameter
tells Scikit-Learn the number of CPU cores to use for training and predictions

'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                           n_estimators = 500,
                           max_samples = 100,bootstrap = True, n_jobs = -1)
#here -1 tells the scikit learn to use all available core

bag_clf.fit(X_train,y_train)
y_pred = bag_clf.predict(X_test)
    
print('Accuracy score after bagging\n',accuracy_score(y_test,y_pred))

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)

print('Accuracy score in Decision Tree classifier\n',accuracy_score(y_test,y_pred))


from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['yellow','blue','green'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['gray','blue','red'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.show()

'''
Out-of-Bag Evaluation

set oob_score=True when creating a BaggingClassifier to request an automatic oob evaluation after training.
'''
bag_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 500,
                            bootstrap = True,n_jobs = -1, oob_score=True)
bag_clf.fit(X_train,y_train)
print(bag_clf.oob_score_)

y_pred = bag_clf.predict(X_test)
print('accuracy score in DTclasifier and baggingclassfier')
accuracy_score(y_test,y_pred)


'''The oob decision function for each training instance is also available through the
oob_decision_function_ variable. In this case (since the base estimator has a predict_proba() 
method) the decision function returns the class probabilities for each
training instance. For example, the oob evaluation estimates that the first training
instance has a 58.05% probability of belonging to the positive class (and 41.95% of
belonging to the negative class):
    [0.41954023 0.58045977]
 [0.33684211 0.66315789]
    
    '''
print(bag_clf.oob_decision_function_)

'''Random Forest is an ensemble of Decision Trees, generally trained via the bagging method (or sometimes pasting), typically with max_samples
set to the size of the training set. Instead of building a BaggingClassifier and passing it a DecisionTreeClassifier, you can instead use the RandomForestClassifier
class, which is more convenient and optimized for Decision Trees10 (similarly, there is a RandomForestRegressor class for regression tasks). The following code trains a
Random Forest classifier with 500 trees (each limited to maximum 16 nodes), using all available CPU cores:'''

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print('Accracy score in random forest \n',accuracy_score(y_test,y_pred))




























