import numpy as np
from sklearn import (datasets, feature_extraction, linear_model, metrics)
import pandas as pd
from sklearn import svm
from sklearn import (tree, ensemble)
from sklearn.neural_network import MLPClassifier

#X_train, X_test = vectorizor(res, res_test)
models = [
    {'name': 'linear_model.SGD', 'obj': linear_model.SGDClassifier()},    82.68
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True)},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[10], solver='adam',n_iter_no_change=5)},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[16], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[64], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[128], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[256], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[512], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[100,100], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[128,128], solver='adam')},
    {'name': 'neural_network.MLP', 'obj': MLPClassifier(verbose=True, early_stopping=True,hidden_layer_sizes=[512,512], solver='adam')},
    {'name': 'svm.LinearSVC', 'obj': svm.LinearSVC()},   0.8358339721826744
    {'name': 'svm.SVC(linear)', 'obj': svm.SVC(kernel='linear')},   0.8351768700032856
    {'name': 'svm.SVC(poly,2)', 'obj': svm.SVC(kernel='poly', degree=2)},   0.8155733216515169
    {'name': 'svm.SVC(poly,3)', 'obj': svm.SVC(kernel='poly')},
    {'name': 'svm.SVC(poly,4)', 'obj': svm.SVC(kernel='poly', degree=4)},
    {'name': 'svm.SVC(rbf)', 'obj': svm.SVC(kernel='rbf')},  0.8247727521629613
    {'name': 'svm.SVC(rbf,$\gamma$=1)', 'obj': svm.SVC(kernel='rbf', gamma=1)}, #82.4
    {'name': 'svm.SVC(rbf,$\gamma$=4)', 'obj': svm.SVC(kernel='rbf', gamma=4)},
    {'name': 'svm.SVC(rbf,$\gamma$=16)', 'obj': svm.SVC(kernel='rbf', gamma=16)},
    {'name': 'svm.SVC(rbf,$\gamma$=64)', 'obj': svm.SVC(kernel='rbf', gamma=64)},
    {'name': 'svm.SVC(sigmoid)', 'obj': svm.SVC(kernel='sigmoid')},
    {'name': 'tree.ExtraTree', 'obj': tree.ExtraTreeClassifier()},
    {'name': 'ensemble.RandomForest(10)', 'obj': ensemble.RandomForestClassifier(n_estimators=10)},
    {'name': 'ensemble.RandomForest(100)', 'obj': ensemble.RandomForestClassifier()},
    {'name': 'ensemble.ExtraTrees(10)', 'obj': ensemble.ExtraTreesClassifier(n_estimators=10)},
    {'name': 'ensemble.ExtraTrees(100)', 'obj': ensemble.ExtraTreesClassifier(n_estimators=100)},
    {'name': 'ensemble.AdaBoost(DTree)', 'obj': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier())},
]

for model in models:
    # Train a model and evaluate it
    print(model['name'])
    model_c = train(X_train, y_train, model['obj'])
    print(test(model_c, X_test, y_test))
    print('='*100)