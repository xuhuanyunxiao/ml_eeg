# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:00:41 2016

@author: yishikeji-01
"""

def py35_ML_SVM(MLdata,MLlabel):
    
#    global ML
    
    #%% 
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time

    
    MLdata = FeatureData
    MLlabel = FeatureLabel
    
    # for reproducibility of the results by explicitly seeding
    random_state = np.random.RandomState(0)
    #%% 1 split train_set and test_set
    Train_Set, Test_Set, Train_Label, Test_label = train_test_split(MLdata,\
          MLlabel, test_size=0.1, random_state=0)
        
    #%% 2 Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()

    # parameter space
    param_grid = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], # 'precomputed'
                  'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}]

    clf = GridSearchCV(SVC(), param_grid,cv = 5)
    clf = clf.fit(Train_Set, Train_Label[0])
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    
    
    #%% Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    Predict_Label = clf.predict(Test_Set)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(Test_label, Predict_Label))
    print(confusion_matrix(Test_label, Predict_Label, labels=range(1,4)))
    
    #%% Qualitative evaluation of the predictions using matplotlib
    


    return

#%%
#if __name__ == '__main__':
#    py35_ML_SVM()
    
    