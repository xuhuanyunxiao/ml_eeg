# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:10:08 2016

@author: yishikeji-01
"""


# 1 K-fold cross validation
# 2 Automatic cross validation
# 3 Cross validation with ShuffleSplit
# 4 Stratified k-fold
# 5 Poor man's grid search
# 6 Brute force grid search
# 7 Using dummy estimatiors to compare results
# 8 Regression model evaluation
# 9 Feature selection
# 10 Feature selection on L1 norms
# 11 Persisting models with joblib

#%% 1 K-fold cross validation
from sklearn.datasets import make_regression
from sklearn.cross_validation import KFold

import numpy as np
import pandas as pd

# how to do it......
## create some fake data
N = 1000
holdout = 200
x,y = make_regression(1000,shuffle = True)

## examine the parameters
### hold out 200 points
x_h,y_h = x[:holdout],y[:holdout]
x_t,y_t = x[holdout:],y[holdout:]
### create the cross validation object
kfold = KFold(len(y_t),n_folds = 4)

## look at the size of the resulting of dataset
### iterate through the k-fold object
output_string = 'Fold:{},N_train:{},N_test:{}'
for i,(train,test) in enumerate(kfold):
    print(output_string.format(i,len(y_t[train]),len(y_t[test])))

    
# how it works......
## create dataset
patients = np.repeat(np.arange(0,100,dtype = np.int8),8)
measurements = pd.DataFrame({'patient_id':patients,
                             'ys':np.random.normal(0,1,800)})

## hold out certain customers instead of data points
custids = np.unique(measurements.patient_id)
customer_kfold = KFold(custids.size,n_folds = 4)
output_string = 'Fold:{},N_train:{},N_test:{}'
for i,(train,test) in enumerate(customer_kfold):
    train_cust_ids = custids[train]
    training = measurements[measurements.patient_id.isin(train_cust_ids)]
    testing = measurements[~measurements.patient_id.isin(train_cust_ids)]
    print(output_string.format(i,len(training),len(testing)))                   

#%% 2 Automatic cross validation
from sklearn import ensemble
from sklearn import datasets
from sklearn import cross_validation

# how to do it......
## create a sample classifier: random forest
rf = ensemble.RandomForestRegressor(max_features = 'auto')

## create some regression data
x,y = datasets.make_regression(10000,10)

## cross_validation
scores = cross_validation.cross_val_score(rf,x,y)
print(scores)


# how it works......
##
scores = cross_validation.cross_val_score(rf,x,y,verbose = 3)
print(scores)
## during each iteration, we can scored the function, like:
##     how long the model runs
## how to create our own scoring function

#%% 3 Cross validation with ShuffleSplit
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit

import numpy as np
import matplotlib.pyplot as plt


# how to do it......
## creat dataset
true_loc = 1000
true_scale = 10
N = 1000
dataset = np.random.normal(true_loc,true_scale,N)

## plot
f,ax = plt.subplots(figsize = (7,5))
ax.hist(dataset,color = 'k',alpha = .65,histtype = 'stepfilled')
ax.set_title('Histogram of dataset')
f.savefig('Cross-validation.png')

## take the first half of the data and guess the mean
holdout_set = dataset[:500]
fitting_set = dataset[500:]
estimate = fitting_set[:N/2].mean()

## plot
f,ax = plt.subplots(figsize = (7,5))
ax.set_title('True Mean vs Regular Estimate')
ax.vlines(true_loc,0,1,color = 'k',linestyle = '-', lw = 5,
          alpha = .65,label = 'true mean')
ax.vlines(estimate,0,1,color = 'g',linestyle = '-', lw = 5,
          alpha = .65,label = 'regular estimate')
ax.set_xlim(999,1001)
ax.legend()
f.savefig('mean.png')

## use ShuffleSplit to fit the estimator on several smaller datasets
shuffle_split = ShuffleSplit(len(fitting_set))
mean_p = []
for train,_ in shuffle_split:
    mean_p.append(fitting_set[train].mean())
    shuf_estimate = np.mean(mean_p)
    
## plot
f,ax = plt.subplots(figsize = (7,5))
ax.set_title('All Estimates')
ax.vlines(true_loc,0,1,color = 'r',linestyle = '-', lw = 5,
          alpha = .65,label = 'true mean')
ax.vlines(estimate,0,1,color = 'g',linestyle = '-', lw = 5,
          alpha = .65,label = 'regular estimate')
ax.vlines(shuf_estimate,0,1,color = 'b',linestyle = '-', lw = 5,
          alpha = .65,label = 'shufflesplit estimate')
ax.set_xlim(999,1001)
ax.legend(loc = 3)
f.savefig('estimate.png')

#%% 4 Stratified k-fold
from sklearn import datasets
from sklearn import cross_validation

import numpy as np
import matplotlib.pyplot as plt
#import itertools as it

# how to do it......
## ceate dataset
x,y = datasets.make_classification(n_samples = int(1e3),
                                   weights = [1./11])
print(y.mean())

## create a stratified k-fold object and iterate it through each fold
n_folds = 50
strat_kfold = cross_validation.StratifiedKFold(y,n_folds = n_folds)
shuff_split = cross_validation.ShuffleSplit(n = len(y),n_iter = n_folds)
kfold_y_props = []
shuff_y_props = []
for (k_train,k_test),(s_train,s_test) in zip(strat_kfold,shuff_split):
    kfold_y_props.append(y[k_train].mean())
    shuff_y_props.append(y[s_train].mean())

## plot
f,ax = plt.subplots(figsize = (7,5))
ax.plot(range(n_folds),shuff_y_props,label = 'ShuffleSplit',color = 'k')
ax.plot(range(n_folds),kfold_y_props,label = 'Stratified',color = 'r',
        ls = '--')
ax.set_title('Comparing class proportions')
ax.legend(loc = 'best')


# how it works......
## getting the overall proportion of the classes
three_classes = np.random.choice([1,2,3],p = [.1,.4,.5],size = 1000)

## intelligently splitting the training and test set into proportions
for train,test in cross_validation.StratifiedKFold(three_classes,5):
    print(np.bincount(three_classes[train]))

#%% 5 Poor man's grid search
from sklearn import datasets
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools as it


# how to do it......
## create the dataset
x,y = datasets.make_classification(n_samples = 2000,n_features = 10)

## represent as python sets
criteria = {'gini','entropy'}
max_features = {'auto','log2',None}
parameter_space = it.product(criteria,max_features)

## iterate through parameter space and check the accuracy of each model
train_set = np.random.choice([True,False],size = len(y))
accuracies = {}
for criterion,max_feature in parameter_space:
    dt = DecisionTreeClassifier(criterion = criterion,
                                max_features = max_feature)
    dt.fit(x[train_set],y[train_set])
    accuracies[(criterion,max_feature)] = (dt.predict(x[~train_set]) 
                                           == y[~train_set]).mean()
print(accuracies)

## visualize the performance
cmap = cm.RdBu_r
f,ax = plt.subplots(figsize = (7,5))
ax.set_xticklabels([''] + list(criteria))
ax.set_yticklabels([''] + list(max_features))
plot_array = []
for max_feature in max_features:
    m = []
for criterion in criteria:
    m.append(accuracies[(criterion,max_feature)])
    plot_array.append(m)
colors = ax.matshow(plot_array,vmin = np.min(accuracies.values()) - 0.001,
            vmax = np.max(accuracies.values()) + 0.001,cmap = cmap)
f.colorbar(colors)

# how it works......
## 1 choose a set of parameters
## 2 iterate through them and find the accuracy of each step
## 3 find the best performer by visual inspection

#%% 6 Brute force grid search
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV

import scipy.stats as st
import numpy as np

# how to do it......
## create some classification data
x,y = make_classification(1000,n_features = 5)

## create logistic regression object
lr = LogisticRegression(class_weight = 'auto')

## specify the parameters we want to search
### GridSearch: specify the ranges 
### RandomizedSearchCV: specify the distribution over the same space from which to sample
lr.fit(x,y)
grid_search_params = {'penalty':['l1','l2'],
                      'C':[1,2,3,4]}
random_search_params = {'penalty':['l1','l2'],
                      'C':st.randint(1,4)}

# how it works......
## fit the classifier: pass lr to the parameters search objects
gs_grid = GridSearchCV(lr,grid_search_params)
gs_grid.fit(x,y)
gs_random = RandomizedSearchCV(lr,random_search_params)
gs_random.fit(x,y)

## access the scores
print(gs_grid.grid_scores_)
print(gs_grid.grid_scores_[1][1])
print(max(gs_grid.grid_scores_,key = lambda x: x[1]))
print(gs_random.grid_scores_)
print(gs_random.grid_scores_[1][1])
print(max(gs_random.grid_scores_,key = lambda x: x[1]))

#%% 7 Using dummy estimatiors to compare results
from sklearn.datasets import make_classification,make_regression
from sklearn import dummy
from sklearn.metrics import accuracy_score

# how to do it......
## create some random data
x,y = make_regression()
dumdum = dummy.DummyRegressor()

## fit various dummy estimators
dumdum.fit(x,y)
dumdum.predict(x)[:5]

### predict a supplied constant (constant = None)
### predict the median value
predictors = [('mean',None),
              ('median',None),
              ('constant',10)]
for strategy,constant in predictors: 
    dumdum = dummy.DummyRegressor(strategy = strategy,
                                  constant = constant)        
dumdum.fit(x,y)
print('strategy:{}'.format(strategy),','.join(map(str,dumdum.predict(x)[:5])))

### four options for classifiers
predictors = [('constant',0),
              ('stratified',None),
              ('uniform',None),
              ('most_frequent',None)]
x,y = make_classification()
for strategy,constant in predictors:
    dumdum = dummy.DummyClassifier(strategy = strategy,
                                   constant = constant)
    dumdum.fit(x,y)
    print('strategy:{}'.format(strategy),','.join(map(str,dumdum.predict(x)[:5])))

# how it works......
## class imbalance causes problems
x,y = make_classification(20000,weights = [.95,.05])
dumdum = dummy.DummyClassifier(strategy = 'most_frequent')
dumdum.fit(x,y)
print(accuracy_score(y,dumdum.predict(x)))
### this is our baseline
### if we cannot create a model for fraud that is more accurate than this,
### then it isn't worth our time

#%% 8 Regression model evaluation
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

from functools import partial

# how to do it......
## 1 use 'y' to generate 'y_actual'
## 2 use 'y_actual' plus some err to generate 'y_prediction'
def data(x,m = 2,b = 1,e = None,s = 10):
    '''
    Args:
        x: The x value
        m: Slope
        b: Intercept
        e: Error, optional, True will give random error
    '''
    if e is None:
        e_i = 0
    elif e is True:
        e_i = np.random.normal(0,s,len(x))
    else:
        e_i = e
        
    return x * m + b + e_i

### define y_hat and y_actual
N = 100
xs = np.sort(np.random.rand(N)*100)
y_pred_gen = partial(data,x = xs,e = True)
y_true_gen = partial(data,x = xs)
y_pred = y_pred_gen()
y_true = y_true_gen()

## plot
f,ax = plt.subplots(figsize = (7,5))
ax.set_title('Plotting the fit vs the underlying process')
ax.scatter(xs,y_pred,label = r'$\hat{y}$')
ax.plot(xs,y_true,label = r'$y$')
ax.legend(loc = 'best')

## walk through various metrics and plot some of them
e_hat = y_pred - y_true
f,ax = plt.subplots(figsize = (7,5))
ax.set_title('Residulas')
ax.hist(e_hat,color = 'k',alpha = .5, histtype = 'stepfilled')


# how it works......
## MSE: find value of the mean squared error
metrics.mean_squared_error(y_true,y_pred)

# MAD: the mean absolute deviation
metrics.r2_score(y_true,y_pred)

#%% 9 Feature selection
from sklearn import datasets
from sklearn import feature_selection

import numpy as np
import matplotlib.pyplot as plt

# how to do it......
## create data
x,y = datasets.make_regression(1000,10000) # 1000 points, 10000 features

## 
f,p = feature_selection.f_regression(x,y)
### f: f score associated with each linear model fit with just one of features
### p: p value associated with that f value
### here: f value is the test statistic

## choose all the p values less than 0.05
idx = np.arange(0,x.shape[1])
features_to_keep = idx[p < .05]
len(features_to_keep)

## set the threshold for which we eliminate features
var_threshold = feature_selection.VarianceThreshold(np.median(np.var(x,axis = 1)))
print(var_threshold.fit_transform(x).shape)


# how it works......
## 
x,y = datasets.make_regression(10000,20)
f,p = feature_selection.f_regression(x,y)

## plot the p values of the features
f,ax = plt.subplots(figsize = (7,5))
ax.bar(np.arange(20),p,color = 'k')
ax.set_title('Feature p values')

#%% 10 Feature selection on L1 norms
import sklearn.datasets as ds
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn import feature_selection

import numpy as np

# how to do it......
## 1 get the dataset
diabetes = ds.load_diabetes()

## 2 create the LinearRegression object
lr = linear_model.LinearRegression()

## 3 the ShuffleSplit cross validation scheme
shuff = cross_validation.ShuffleSplit(diabetes.target.size)

## 4 fit the model, and keep the track of MSE for each iteration of ShuffleSplit
mses = []
for train,test in shuff:
    train_x = diabetes.data[train]
    train_y = diabetes.target[train]
    
    test_x = diabetes.data[~train]
    test_y = diabetes.target[~train]

    lr.fit(train_x,train_y)
    
    mses.append(metrics.mean_squared_error(test_y,lr.predict(test_x)))

print(np.mean(mses))

## 5 fit the Lasso Regression
cv = linear_model.LassoCV()
cv.fit(diabetes.data,diabetes.target)
print(cv.coef_)

## 6 check it after eliminate any features with a zero for the coefficient
columns = np.arange(diabetes.data.shape[1])[cv.coef_ != 0]
print(columns)
 
## 7 fit the model with the specific features
l1mses = []
for train,test in shuff:
    train_x = diabetes.data[train][:,columns]
    train_y = diabetes.target[train]
    
    test_x = diabetes.data[~train][:,columns]
    test_y = diabetes.target[~train]

    lr.fit(train_x,train_y)
    
    l1mses.append(metrics.mean_squared_error(test_y,lr.predict(test_x)))
 
print(np.mean(l1mses))
print(np.mean(l1mses) - np.mean(mses))


# how it works......
## 1 create a regression dataset with many uninformative features
x,y = ds.make_regression(noise = 5)

## 2 fit a normal regression
shuff = cross_validation.ShuffleSplit(y.size)
mses = []
for train,test in shuff:
    train_x = x[train]
    train_y = y[train]
    
    test_x = x[~train]
    test_y = y[~train]

    lr.fit(train_x,train_y)
    
    mses.append(metrics.mean_squared_error(test_y,lr.predict(test_x)))

print(np.mean(mses))

## 3 walk through the same process for Lasso regression
cv.fit(x,y)
columns = np.arange(x.shape[1])[cv.coef_ != 0]
print(columns)
shuff = cross_validation.ShuffleSplit(y.size)
mses = []
for train,test in shuff:
    train_x = x[train][:,columns]
    train_y = y[train]
    
    test_x = x[~train][:,columns]
    test_y = y[~train]

    lr.fit(train_x,train_y)
    
    mses.append(metrics.mean_squared_error(test_y,lr.predict(test_x)))

print(np.mean(mses))

#%% 11 Persisting models with joblib
from sklearn import datasets,tree
from sklearn import ensemble
from sklearn.externals import joblib

# how to do it......
x,y = datasets.make_classification()
dt = tree.DecisionTreeClassifier()
dt.fit(x,y)

joblib.dump(dt,'dtree.clf')
clf_DT = joblib.load('dtree.clf') 

# how it works......
rf = ensemble.RandomForestClassifier()
rf.fit(x,y)

## omit the output, but in total, there we were 52 files outputted on my machine
joblib.dump(rf, "rf.clf")
clf_RFC = joblib.load('dtree.clf') 






