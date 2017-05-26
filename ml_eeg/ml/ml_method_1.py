# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:00:07 2017

@author: yishikeji-01
"""

import numpy as np
import pandas as pd

###############################################################################
#-------  交叉验证与参数寻优  --------------------------------------------------
###############################################################################

def cross_validation(train_set,train_target,clf,param_grid):
    from sklearn.model_selection import GridSearchCV    
    
    nfolds = 5
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
    
    return best_clf

def get_ml(train_set,train_target,test_set,test_target,ml_method):
    if ml_method == 'SVM':
        from sklearn.svm import SVC    
        # parameters search and cross-validation
        kernels = ['rbf']
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'kernel':kernels,'C': Cs, 'gamma' : gammas}
        clf = SVC(random_state = 10)
    elif ml_method == 'BP':
        ml_result = ml_bp(train_set,train_target,test_set,test_target)
    elif ml_method == 'DecisionTree':
        ml_result = ml_Dtree(train_set,train_target,test_set,test_target)
    elif ml_method == 'NaiveBayes':
        ml_result = ml_nb(train_set,train_target,test_set,test_target)
    elif ml_method == 'KNN':
        ml_result = ml_knn(train_set,train_target,test_set,test_target)
    elif ml_method == 'Logistic':
        ml_result = ml_logistic(train_set,train_target,test_set,test_target)
    elif ml_method == 'RandomTree':        
        ml_result = ml_Rtree(train_set,train_target,test_set,test_target)
    elif ml_method == 'GDBT':        
        ml_result = ml_gdbt(train_set,train_target,test_set,test_target)
    elif ml_method == 'Adaboosting':        
        ml_result = ml_adaboost(train_set,train_target,test_set,test_target)
    elif ml_method == 'Boosted':        
        ml_result = ml_boost(train_set,train_target,test_set,test_target)
    elif ml_method == 'BaggedTrees':        
        ml_result = ml_bag(train_set,train_target,test_set,test_target)
        
    best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target
        
###############################################################################
#-------  单一方法  ------------------------------------------------------------
###############################################################################

def ml_svm(train_set,train_target,test_set,test_target):
    from sklearn.svm import SVC
    
    # parameters search and cross-validation
    kernels = ['rbf']
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'kernel':kernels,'C': Cs, 'gamma' : gammas}
    clf = SVC(random_state = 10)
    
    best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target
    
def ml_bp(train_set,train_target,test_set,test_target):
    from sklearn.neural_network import MLPClassifier
    
    # parameters search and cross-validation
    hidden_layer_sizes  = np.arange(5,41)
    activation  = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.0001, 0.001, 0.01, 0.1]
    param_grid = {'hidden_layer_sizes':hidden_layer_sizes,
                  'activation': activation, 
                  'solver': solver,
                  'alpha' : alpha}
    clf = MLPClassifier(random_state = 10)
    
    best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target     
    
def ml_Dtree(train_set,train_target,test_set,test_target):
    from sklearn.tree import DecisionTreeClassifier
    
    # parameters search and cross-validation
    criterion  = ['gini','entropy']
    max_depth = np.arange(10,100,10)
    min_samples_leaf = [1,2,3,4,5]
    # class_weight = ['None','Banlanced']
    param_grid = {'criterion':criterion,
                  'max_depth': max_depth, 
                  'min_samples_leaf': min_samples_leaf}
    clf = DecisionTreeClassifier(random_state = 10)
    
    best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target   

def ml_nb(train_set,train_target,test_set,test_target):
    from sklearn.naive_bayes import GaussianNB
    
    # parameters search and cross-validation
    kernels = ['rbf']
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'kernel':kernels,'C': Cs, 'gamma' : gammas}
    clf = GaussianNB(random_state = 10)
    
    best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target
    
def ml_knn(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier   
    
    # parameters search and cross-validation
    n_neighbor = np.arange(5,41)
    weight = ['uniform','distance']
    algorithm = ['ball_tree','kd_tree','brute']
    param_grid = {'n_neighbors':n_neighbor,'weights': weight, 'algorithm' : algorithm}
    
    clf = KNeighborsClassifier(random_state = 10)
    best_clf = cross_validation(train_set,train_target,clf,param_grid)

    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target    
    
def ml_logistic(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier
    
    # parameters search and cross-validation
    kernels = ['rbf']
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'kernel':kernels,'C': Cs, 'gamma' : gammas}
    clf = SVC(random_state = 10)
    
    best_clf = cross_validation(train_set,train_target,clf,param_grid)
    
    # ml
    best_clf.fit(train_set,train_target)  # train          
    predict_target = best_clf.predict(test_set)  # test
    
    return best_clf, predict_target
    
###############################################################################
#-------  组合方法  ------------------------------------------------------------
###############################################################################

def ml_Rtree(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier
    
    # cross-validation
    n_neighbor = np.arange(5,40)
    weight = ['uniform','distance']
    algorithm = ['ball_tree','kd_tree','brute']
    param_grid = {'n_neighbor':n_neighbor,'weight': weight, 'algorithm' : algorithm}
    
    clf = KNeighborsClassifier()
    best_params = cross_validation(train_set,train_target,clf,param_grid)

    # ml
    clf = KNeighborsClassifier(best_params)
    clf.fit(train_set,train_target)  # train          
    clf.predict(test_set)  # test
    
    # evaluation  
    
def ml_gdbt(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier
    
    # cross-validation
    n_neighbor = np.arange(5,40)
    weight = ['uniform','distance']
    algorithm = ['ball_tree','kd_tree','brute']
    param_grid = {'n_neighbor':n_neighbor,'weight': weight, 'algorithm' : algorithm}
    
    clf = KNeighborsClassifier()
    best_params = cross_validation(train_set,train_target,clf,param_grid)

    # ml
    clf = KNeighborsClassifier(best_params)
    clf.fit(train_set,train_target)  # train          
    clf.predict(test_set)  # test
    
    # evaluation  
    

def ml_adaboost(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier
    
    # cross-validation
    n_neighbor = np.arange(5,40)
    weight = ['uniform','distance']
    algorithm = ['ball_tree','kd_tree','brute']
    param_grid = {'n_neighbor':n_neighbor,'weight': weight, 'algorithm' : algorithm}
    
    clf = KNeighborsClassifier()
    best_params = cross_validation(train_set,train_target,clf,param_grid)

    # ml
    clf = KNeighborsClassifier(best_params)
    clf.fit(train_set,train_target)  # train          
    clf.predict(test_set)  # test
    
    # evaluation  
    

def ml_boost(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier
    
    # cross-validation
    n_neighbor = np.arange(5,40)
    weight = ['uniform','distance']
    algorithm = ['ball_tree','kd_tree','brute']
    param_grid = {'n_neighbor':n_neighbor,'weight': weight, 'algorithm' : algorithm}
    
    clf = KNeighborsClassifier()
    best_params = cross_validation(train_set,train_target,clf,param_grid)

    # ml
    clf = KNeighborsClassifier(best_params)
    clf.fit(train_set,train_target)  # train          
    clf.predict(test_set)  # test
    
    # evaluation  
    

def ml_bag(train_set,train_target,test_set,test_target):
    from sklearn.neighbors import KNeighborsClassifier
    
    # cross-validation
    n_neighbor = np.arange(5,40)
    weight = ['uniform','distance']
    algorithm = ['ball_tree','kd_tree','brute']
    param_grid = {'n_neighbor':n_neighbor,'weight': weight, 'algorithm' : algorithm}
    
    clf = KNeighborsClassifier()
    best_params = cross_validation(train_set,train_target,clf,param_grid)

    # ml
    clf = KNeighborsClassifier(best_params)
    clf.fit(train_set,train_target)  # train          
    clf.predict(test_set)  # test
    
    # evaluation  
    