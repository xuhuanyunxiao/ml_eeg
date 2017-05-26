# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:12:15 2016

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

import quandl,math
import datetime
import pickle

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression



#%% 2 Regression - intro and data
df = quandl.get("WIKI/GOOGL")
print(df.head())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

#%% 3 Regression - features and labels
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) # deal with missing data
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())

df.dropna(inplace = True)
print(df.tail())

#%% 4 Regression - training and testing
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
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

#%% 5 Regression - forecasting and predicting
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)

style.use('ggplot')
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

#%% 6 Regression - picking and scaling
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)



















#%% 7 Regression - theory and how it works

#%% 8 Regression - how to program the best fit slope

#%% 9 Regression - how to program the best fit line


#%% 10 Regression - R squared and coefficient of determination theory


#%% 11 Regression - how to program R squared

#%% 12 Regression - ceating sample data for testing

#%% 13 KNN - intro with K nearest neighbors


#%% 14 KNN - applying K nearest neighbors to data

#%% 15 KNN - euclidean distance theory

#%% 16 KNN - creating a K nearest neighbors classifier from scrath

#%% 17 KNN - creating a K nearest neighbors classifier from scrath part 2


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

