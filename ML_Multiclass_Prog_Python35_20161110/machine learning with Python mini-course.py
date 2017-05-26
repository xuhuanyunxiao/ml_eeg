# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:18:45 2016

@author: yishikeji-01
"""
#%% Table of content
# Day 1: Download and Install Python and SciPy Ecosystem
# Day 2: Try Some Basic Python and SciPy Syntax
# Day 3: Load Datasets from CSV
# Day 4: Understand Data with Descriptive Stats
# Day 5: Understand Data with Data Visualization
# Day 6: Prepare For Modeling by Pre-Processing Data
# Day 7: Algorithm Evaluation With Resampling Methods
# Day 8: Algorithm Evaluation Metrics
# Day 9: Spot-Check Machine Learning Algorithms
# Day 10: Model Comparison and Selection
# Day 11: Improve Accuracy with Algorithm Tuning
# Day 12: Improve Accuracy with Ensemble Predictions
# Day 13: Finalize And Save Your Model
# Day 14: Hello World End-to-End Project

#%% Day 1: Download and Install Python and SciPy Ecosystem
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
#%% Day 2: Try Some Basic Python and SciPy Syntax
# uses hash (#) for comments 
# uses white space to indicate code blocks (white space matters).

#Practice assignment, working with lists and flow control in Python.
#Practice working with NumPy arrays.
#Practice creating simple plots in Matplotlib.
#Practice working with Pandas Series and DataFrames.

# dataframe
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

#%% Day 3: Load Datasets from CSV

#download and practice with on the UCI machine learning repository.

#Practice loading CSV files into Python using the CSV.reader() function in the standard library.
#Practice loading CSV files using NumPy and the numpy.loadtxt() function.
#Practice loading CSV files using Pandas and the pandas.read_csv() function.

# Load CSV using Pandas from URL
from pandas import read_csv
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
print(data.shape)

#%% Day 4: Understand Data with Descriptive Stats

#Understand your data using the head() function to look at the first few rows.
#Review the dimensions of your data with the shape property.
#Look at the data types for each attribute with the dtypes property.
#Review the distribution of your data with the describe() function.
#Calculate pair-wise correlation between your variables using the corr() function.

# Statistical Summary
import pandas
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
description = data.describe()
print(description)

#%% Day 5: Understand Data with Data Visualization

#Use the hist() function to create a histogram of each attribute.
#Use the plot(kind='box') function to create box and whisker plots of each attribute.
#Use the pandas.scatter_matrix() function to create pair-wise scatter plots of all attributes.

# Scatter Plot Matrix
import matplotlib.pyplot as plt
import pandas
from pandas.tools.plotting import scatter_matrix
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
scatter_matrix(data)
plt.show()

#%% Day 6: Prepare For Modeling by Pre-Processing Data

#Standardize numerical data (e.g. mean of 0 and standard deviation of 1) using the scale and center options.
#Normalize numerical data (e.g. to a range of 0-1) using the range option.
#Explore more advanced feature engineering such as Binarizing.

# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])

#%% Day 7: Algorithm Evaluation With Resampling Methods

#Split a dataset into training and test sets.
#Estimate the accuracy of an algorithm using k-fold cross validation.
#Estimate the accuracy of an algorithm using leave one out cross validation.

# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#%% Day 8: Algorithm Evaluation Metrics

#Practice using the Accuracy and Kappa metrics on a classification problem.
#Practice generating a confusion matrix and a classification report.
#Practice using RMSE and RSquared metrics on a regression problem.

# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("Logloss: %.3f (%.3f)") % (results.mean(), results.std())​)

#%% Day 9: Spot-Check Machine Learning Algorithms

#Spot-check linear algorithms on a dataset (e.g. linear regression, logistic regression and linear discriminate analysis).
#Spot-check some nonlinear algorithms on a dataset (e.g. KNN, SVM and CART).
#Spot-check some sophisticated ensemble algorithms on a dataset (e.g. random forest and stochastic gradient boosting).

# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#%% Day 10: Model Comparison and Selection

#Compare linear algorithms to each other on a dataset.
#Compare nonlinear algorithms to each other on a dataset.
#Create plots of the results comparing algorithms.

# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load dataset
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#%% Day 11: Improve Accuracy with Algorithm Tuning

#The scikit-learn library provides two ways to search for combinations of parameters for a machine learning algorithm:
#Tune the parameters of an algorithm using a grid search that you specify.
#Tune the parameters of an algorithm using a random search.

# Grid Search for Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

#%% Day 12: Improve Accuracy with Ensemble Predictions

#Practice bagging ensembles with the Random Forest and Extra Trees algorithms.
#Practice boosting ensembles with the Gradient Boosting Machine and AdaBoost algorithms.
#Practice voting ensembles using by combining the predictions from multiple models together.

# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#%% Day 13: Finalize And Save Your Model

#Practice making predictions with your model on new data (data unseen during training and testing).
#Practice saving trained models to file and loading them up again.

# Save Model Using Pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

#%% Day 14: Hello World End-to-End Project

#Understanding your data using descriptive statistics and visualization.
#Pre-Processing the data to best expose the structure of the problem.
#Spot-checking a number of algorithms using your own test harness.
#Improving results using algorithm parameter tuning.
#Improving results using ensemble methods.
#Finalize the model ready for future use.

