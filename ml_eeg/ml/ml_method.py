# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:19:17 2017

@author: yishikeji-01
"""

#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

#%%
clf_set = [SVC(random_state = 10, probability=True),
           MLPClassifier(random_state = 10),
           DecisionTreeClassifier(random_state = 10),
           GaussianNB(),
           MultinomialNB(),
           KNeighborsClassifier(),
           LogisticRegression(random_state=10),
           RandomForestClassifier(random_state=10),
           GradientBoostingClassifier(random_state = 10)]

names = ['svc','bp','dt','gnb','mnb','knn','lr','rf','gbc']
nfolds = 5

#%%
def cross_validation_nb(train_set,train_target):    
    # GaussianNB
    gnb_clf = clf_set[3]
    gnb_score = max(cross_val_score(gnb_clf, train_set, train_target, cv = nfolds))
    # MultinomialNB                
    alpha = [0.001, 0.01, 0.1, 0.3, 0.6, 1]
    param_grid = {'alpha':alpha}
    mnb_clf = clf_set[4]         
    grid_search = GridSearchCV(mnb_clf, param_grid, cv=nfolds)
    grid_search.fit(train_set, train_target)
    mnb_score = grid_search.best_score_
    
    if gnb_score > mnb_score:
        best_clf = GaussianNB()
    else:
        best_clf = grid_search.best_estimator_()
    
    return best_clf
    
#%%
def cross_validation_voting(train_set,train_target):
    from itertools import combinations
    score_set = []
    estimators = []
    for name,clf in zip(names,clf_set):estimators.append((name,clf))
    for estimator_n in range(1,10):
        estimator_index = list(combinations([1,2,3,4,5,6,7,8], estimator_n))
        for index in estimator_index:
            estimator = []
            for i in index:estimator.append(estimators[i])
            clf = VotingClassifier(estimator)  # voting = ['hard','soft']
            scores = cross_val_score(clf, train_set, train_target, cv=nfolds, scoring='accuracy')
            score_set.append([index,scores.mean(), scores.std()])
            # print(index)
            # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'voting'))
    score_mean = []
    for i in range(len(score_set)):score_mean.append(score_set[i][1])
    #pd.DataFrame(score_mean).plot()
    max_score_index = max( (v, i) for i, v in enumerate(score_mean) )[1]
    index = score_set[max_score_index][0]
    estimator = []
    for i in index:estimator.append(estimators[i])
    
    best_clf = VotingClassifier(estimator)
    
    return best_clf    
    
#%%
def cross_validation(train_set,train_target,clf,param_grid):          
    grid_search = GridSearchCV(clf, param_grid, cv=nfolds)
    grid_search.fit(train_set, train_target)
    
    best_clf = grid_search.best_estimator_
    # grid_search.grid_scores_
    # cv_results_  best_params_ best_score_  n_splits_  grid_scores_  best_index_
    
    # plot
    if 0:
        b = grid_search.grid_scores_
        c = []
        for i in range(len(b)):c.append(b[i][1])
        pd.DataFrame(c).plot()
        
        #print(accuracy_score(test_target, predict_target))
        #print(classification_report(test_target, predict_target))
        
    return best_clf

#%%    
def get_ml(train_set,train_target,test_set,test_target,ml_method):
    
    #-------  单一方法  ---------------------------------------------
    if ml_method == 'SVM':            
        # parameters search and cross-validation
        kernels = ['rbf']
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'kernel':kernels,'C': Cs, 'gamma' : gammas}
        clf = clf_set[0]
        
    elif ml_method == 'BP':        
        hidden_layer_sizes  = np.arange(5,41)
        activation  = ['identity', 'logistic', 'tanh', 'relu']
        solver = ['lbfgs', 'sgd', 'adam']
        alpha = [0.0001, 0.001, 0.01, 0.1]
        param_grid = {'hidden_layer_sizes':hidden_layer_sizes,
                  'activation': activation, 
                  'solver': solver,
                  'alpha' : alpha}
        clf = clf_set[1]    
    
    elif ml_method == 'DecisionTree':        
        criterion  = ['gini','entropy']
        max_depth = np.arange(10,100,10)
        min_samples_leaf = [1,2,3,4,5]
        # class_weight = ['None','Banlanced']
        param_grid = {'criterion':criterion,
                  'max_depth': max_depth, 
                  'min_samples_leaf': min_samples_leaf}
        clf = clf_set[2]

    # elif ml_method == 'NaiveBayes':
        
    elif ml_method == 'KNN':           
        n_neighbor = np.arange(5,41)
        weight = ['uniform','distance']
        algorithm = ['ball_tree','kd_tree','brute']
        param_grid = {'n_neighbors':n_neighbor,
                'weights': weight, 'algorithm' : algorithm}    
        clf = clf_set[5]

    elif ml_method == 'Logistic':        
        penalty = ['l2']
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        solver = [ 'lbfgs', 'liblinear', 'sag']
        param_grid = {'penalty':penalty,'C': Cs, 'solver' : solver}
        clf = clf_set[6]
        
    #-------  组合方法  ------------------------------------------------------
    elif ml_method == 'RandomTree':                
        criterion  = ['gini','entropy']
        max_depth = np.arange(10,50,10)
        min_samples_leaf = np.arange(1, 50, 10)
        n_estimators = np.arange(1, 200, 20)
        param_grid = {'criterion':criterion,
                  'max_depth': max_depth, 
                  'min_samples_leaf': min_samples_leaf, 
                  'n_estimators' : n_estimators}    
        clf = RandomForestClassifier(random_state = 10)

    elif ml_method == 'GDBT':                
        criterion  = ['gini','entropy']
        max_depth = np.arange(10,50,10)
        min_samples_leaf = np.arange(1, 50, 10)
        n_estimators = np.arange(1, 200, 20)
        param_grid = {'criterion':criterion,
                  'max_depth': max_depth, 
                  'min_samples_leaf': min_samples_leaf, 
                  'n_estimators' : n_estimators}   
        clf = GradientBoostingClassifier(random_state = 10)    
            
    elif ml_method == 'Adaboosting':        
        #base_estimator  = clf_set
        algorithm  = ['SAMME','SAMME.R']
        learning_rate  = [0.001,0.05,0.1,0.3,0.5,0.8,1]
        n_estimators = np.arange(1, 200, 20)
        param_grid = {
                  'algorithm': algorithm, 
                  'learning_rate': learning_rate,
                  'n_estimators' : n_estimators}   
        clf = AdaBoostClassifier(random_state = 10)
        
    elif ml_method == 'BaggedTrees':        
        #base_estimator = clf_set
        n_estimators = np.arange(1, 200, 20)
        param_grid = {
                  'n_estimators' : n_estimators}   
        clf = BaggingClassifier(random_state = 10)
        
    # elif ml_method == 'Voting':
                
        
    if ml_method == 'NaiveBayes':
        best_clf = cross_validation_nb(train_set,train_target)
    elif ml_method == 'Voting':
        best_clf = cross_validation_voting(train_set,train_target)
    else:
        best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    predict_score = best_clf.predict_proba(test_set)
    
    return best_clf, predict_target,predict_score
    
    
    