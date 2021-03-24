import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#On charge nos données depuis le fichier excel
df = pd.read_excel('C:\\Users\\charp\\Downloads\\final_dataset.xlsx')
df.head()

X = df.drop(['alim_grp_code', 'alim_ssgrp_code', 'alim_ssssgrp_code'], axis=1)
Y = df[['alim_grp_code']]
Y = Y.values.ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


from sklearn.linear_model import RidgeClassifier  # 1
from sklearn import svm  # 2
from sklearn.linear_model import SGDClassifier  # 3
from sklearn.neighbors import KNeighborsClassifier  # 4
from sklearn.naive_bayes import GaussianNB  # 5
from sklearn.tree import DecisionTreeClassifier  # 6
from sklearn.ensemble import RandomForestClassifier  # 7
from sklearn.neural_network import MLPClassifier  # 8

from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score

clf_RidgeClassifier = RidgeClassifier()
scores_clf_RidgeClassifier = cross_val_score(clf_RidgeClassifier, X, Y, cv=5, scoring="accuracy")
print("score Ridge : "+str(scores_clf_RidgeClassifier.mean()))
clf_RidgeClassifier.fit(X_train,Y_train)
clf_RidgeClassifier.predict(X_test)

clf_SVM = svm.SVC()
scores_SVM = cross_val_score(clf_SVM, X, Y, cv=5, scoring="accuracy")
print("score SVM : "+str(scores_SVM.mean()))

clf_SGD = SGDClassifier(loss="hinge", penalty="l2", max_iter=500)
clf_DecisionTree = clf_SGD.fit(X, Y)
scores_SGD = cross_val_score(clf_SGD, X, Y, cv=5, scoring="accuracy")
print("score SGD : "+str(scores_SGD.mean()))

from sklearn.model_selection import GridSearchCV

n_neighbors=np.arange(180,200)
weights=np.array(['uniform', 'distance'])

param_grid = {"n_neighbors":n_neighbors,"weights":weights}
grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(X,Y)
print("les meilleurs paramètres sont "+ str(grid.best_params_))
best_params=grid.best_params_

clf_KNeighborsClassifier = KNeighborsClassifier(n_neighbors=best_params.get("n_neighbors"),weights=best_params.get("weights"))
scores_KNeighborsClassifier = cross_val_score(clf_KNeighborsClassifier, X, Y, cv=5, scoring="accuracy")
print("score KNeighborsClassifier : "+str(scores_KNeighborsClassifier.mean()))

gnb = GaussianNB()
gnb = gnb.fit(X, Y)
scores_GaussianNB = cross_val_score(gnb, X, Y, cv=5, scoring="accuracy")
print("score GaussianNB : "+str(scores_GaussianNB.mean()))

clf_DecisionTree = DecisionTreeClassifier()
clf_DecisionTree = clf_DecisionTree.fit(X, Y)
scores_DecisionTree = cross_val_score(clf_DecisionTree, X, Y, cv=5, scoring="accuracy")
print("score DecisionTree : "+str(scores_DecisionTree.mean()))


n_estimators=np.arange(60,61)
max_depth=np.arange(100,101)

param_grid = {"n_estimators":n_estimators,"max_depth":max_depth}
grid = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
grid.fit(X,Y)
print("les meilleurs paramètres sont"+ str(grid.best_params_))
best_params=grid.best_params_

clf_RandomForestClassifier = RandomForestClassifier(max_depth=best_params.get("max_depth"),n_estimators=best_params.get("n_estimators"))
# ou grid.best_estimator_
scores_RandomForestClassifier = cross_val_score(clf_RandomForestClassifier, X, Y, cv=5, scoring="accuracy")
print("score RandomForestClassifier : "+str(scores_RandomForestClassifier.mean()))

# alpha=np.array([10**-k for k in range(1,10)])
# hidden_layer_sizes=np.array([])
# for k in range(1,10):
#     for l in range(1,10):
#         hidden_layer_sizes=np.append(hidden_layer_sizes,(k,l))
param_grid = {
        'hidden_layer_sizes': [(7, 7), (128,128), (128, 7)],
        'alpha': [1e-8]
    }

#param_grid = {"alpha":alpha,"hidden_layer_sizes":hidden_layer_sizes}
grid = GridSearchCV(MLPClassifier(),param_grid,cv=5)
grid.fit(X,Y)
print("les meilleurs paramètres sont"+ str(grid.best_params_))
best_params=grid.best_params_

clf_MLPClassifier = MLPClassifier(solver='lbfgs', alpha=best_params.get("alpha"), hidden_layer_sizes=best_params.get("hidden_layer_sizes"))
scores_MLPClassifier = cross_val_score(clf_MLPClassifier, X, Y, cv=5, scoring="accuracy")
print("score MLPClassifier : "+str(scores_MLPClassifier.mean()))