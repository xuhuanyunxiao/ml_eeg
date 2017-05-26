# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:05:24 2016

@author: yishikeji-01
"""

#%% 1 introduction
# 2 Regression - intro and data
# 3 Regression - features and labels
# 4 Regression - training and testing
# 5 Regression - forecasting and predicting
# 6 Regression - picking and scaling
# 7 Regression - theory and how it works
# 8 Regression - how to program the best fit slope
# 9 Regression - how to program the best fit line
# 10 Regression - R squared and coefficient of determination theory
# 11 Regression - how to program R squared
# 12 Regression - ceating sample data for testing
# 13 KNN - intro with K nearest neighbors
# 14 KNN - applying K nearest neighbors to data
# 15 KNN - euclidean distance theory
# 16 KNN - creating a K nearest neighbors classifier from scrath
# 17 KNN - creating a K nearest neighbors classifier from scrath part 2
# 18 KNN - testing our K nearest neighbors classifier
# 19 KNN - final thoughts on K nearest neighbors
# 20 SVM - support vector machine introduction
# 21 SVM - vector basics 
# 22 SVM - support vector assertions
# 23 SVM - support vector machine fundamentals
# 24 SVM - constraint optimization with support vector machine
# 25 SVM - beginning SVM from scratch in python
# 26 SVM - support vector machine optimization in python 
# 27 SVM - support vector machine optimization in python part2
# 28 SVM - Visualization and predicting with our custom svm
# 29 SVM - kernels introduction
# 30 SVM - why kernels
# 31 SVM - soft margin support vector machine
# 32 SVM - kernels,soft margin svm, and quadratic programming with python and cvxopy
# 33 SVM - support vector machine parameters
# 34 Clustering - introduction
# 35 Clustering - handling non-numerical data for machine learning
# 36 Clustering - K-means with Titanic dataset
# 37 Clustering - K-means from scratch in python
# 38 Clustering - finishing K-means from scratch in python
# 39 Clustering - hierarchical clustering with mean shift introduction
# 40 Clustering - mean shift applied to titanic dataset
# 41 Clustering - mean shift algorithm from scratch in python
# 42 Clustering - dynamically weighted bandwidth for mean shift
# 43 Nerual-Networks - introduction
# 44 Nerual-Networks - installing tensorflow for deep learning
# 45 Deep-Learning - introduction
# 46 Deep-Learning - creating the neural network model
# 47 Deep-Learning - how the network will run
# 48 Deep-Learning - our own data
# 49 Deep-Learning - simple preprocessing language data
# 50 Deep-Learning - training and testing on our data
# 51 Deep-Learning - 10k samples compared to 1.6 million samples
# 52 Deep-Learning - how to use CUDA and GPU version of TensorFolw
# 53 Deep-Learning - RNN basic and LSTM cell
# 54 Deep-Learning - RNN w/ LSTM cell example in TensorFlow and python
# 55 Deep-Learning - CNN basics
# 56 Deep-Learning - CNN
# 57 Deep-Learning - TFlearn- high level abstraction layer



#%%
import quandl, math
import datetime
import pickle
import random
import warnings

from statistics import mean
from math import sqrt
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import svm
#%% 2 Regression - Intro and Data
df = quandl.get("WIKI/GOOGL")
print(df.head())
#print(df.tail())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

#%% 3 Regression - Features and Labels
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

#%% 4 Regression - Training and Testing
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

clf = LinearRegression()
clf = LinearRegression(n_jobs=-1)

for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)

#%% 5 Regression - Forecasting and Predicting
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'][:-forecast_out])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#%% 6 Pickling and Scaling
# save model
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

#%% 8 Regression - How to program the Best Fit Slope
xs = [1,2,3,4,5]
ys = [5,4,6,5,6]

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)


def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)))
    return m

m = best_fit_slope(xs,ys)
print(m)

#%% ##################################
def best_fit_slope(xs,ys):
    m = (mean(xs) * mean(ys))
    return m

def best_fit_slope(xs,ys):
    m = ( (mean(xs)*mean(ys)) - mean(xs*ys) )
    return m    
    
def best_fit_slope(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys)) /
           (mean(xs)**2))
    return m    
    
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)))
    return m
    
#%% 9 Regression - how to program the best fit line
xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)

print(m,b)    
   
regression_line = [(m*x)+b for x in xs]
regression_line = []
for x in xs:
    regression_line.append((m*x)+b)

style.use('ggplot')

plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.show()

predict_x = 7

predict_y = (m*predict_x)+b
print(predict_y)

predict_x = 7
predict_y = (m*predict_x)+b

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()

#%% 11 Regression - how to program R squared
style.use('ggplot')

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
    
m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)        
    
#%% 12 Regression - ceating sample data for testing
style.use('ggplot')

def create_dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)

    return m, b


def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    print(squared_error_regr)
    print(squared_error_y_mean)

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared

xs, ys = create_dataset(40,40,2,correlation='pos')
m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]
r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

plt.scatter(xs,ys,color='#003F72', label = 'data')
plt.plot(xs, regression_line, label = 'regression line')
plt.legend(loc=4)
plt.show()
    
#%% 13 KNN - intro with K nearest neighbors


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
#df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#%%
example_measures = np.array([4,2,1,1,1,2,3,2,1])

prediction = clf.predict(example_measures)
print(prediction)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(2, -1)
prediction = clf.predict(example_measures)
print(prediction)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

#%% 14 KNN - applying K nearest neighbors to data





#%% 15 KNN - euclidean distance theory
plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

#%% 16 KNN - creating a K nearest neighbors classifier from scrath
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)

plt.show()

#%% ############################
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    return vote_result




#%% 17 KNN - creating a K nearest neighbors classifier from scrath part 2
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()      
            
#%% 18 KNN - testing our K nearest neighbors classifier

#%% 19 KNN - final thoughts on K nearest neighbors


#%% 20 SVM - support vector machine introduction

#%% 21 SVM - vector basics 

#%% 22 SVM - support vector assertions

#%% 23 SVM - support vector machine fundamentals

#%% 24 SVM - constraint optimization with support vector machine

#%% 25 SVM - beginning SVM from scratch in python

#%% 26 SVM - support vector machine optimization in python 

#%% 27 SVM - support vector machine optimization in python part2

#%% 28 SVM - Visualization and predicting with our custom svm

#%% 29 SVM - kernels introduction

#%% 30 SVM - why kernels

#%% 31 SVM - soft margin support vector machine

#%% 32 SVM - kernels,soft margin svm, and quadratic programming with python and cvxopy


#%% 33 SVM - support vector machine parameters

#%% 34 Clustering - introduction


#%% 35 Clustering - handling non-numerical data for machine learning

#%% 36 Clustering - K-means with Titanic dataset


#%% 37 Clustering - K-means from scratch in python


#%% 38 Clustering - finishing K-means from scratch in python

#%% 39 Clustering - hierarchical clustering with mean shift introduction

#%% 40 Clustering - mean shift applied to titanic dataset

#%% 41 Clustering - mean shift algorithm from scratch in python


#%% 42 Clustering - dynamically weighted bandwidth for mean shift

#%% 43 Nerual-Networks - introduction


#%% 44 Nerual-Networks - installing tensorflow for deep learning


#%% 45 Deep-Learning - introduction

#%% 46 Deep-Learning - creating the neural network model

#%% 47 Deep-Learning - how the network will run

#%% 48 Deep-Learning - our own data


#%% 49 Deep-Learning - simple preprocessing language data

#%% 50 Deep-Learning - training and testing on our data


#%% 51 Deep-Learning - 10k samples compared to 1.6 million samples


#%% 52 Deep-Learning - how to use CUDA and GPU version of TensorFolw

#%% 53 Deep-Learning - RNN basic and LSTM cell

#%% 54 Deep-Learning - RNN w/ LSTM cell example in TensorFlow and python

#%% 55 Deep-Learning - CNN basics


#%% 56 Deep-Learning - CNN

#%% 57 Deep-Learning - TFlearn- high level abstraction layer


