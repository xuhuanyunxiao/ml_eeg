# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 18:32:58 2016

@author: yishikeji-01
"""

#%% import libraries which will be used
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

import pydot

#%% Decision Tree: import the object, then fit the model

# this was just a dry run
## get some classification data
x,y = datasets.make_classification(n_samples = 1000, n_features = 3,
                                   n_redundant = 0)
dt = DecisionTreeClassifier()
dt.fit(x,y)

preds = dt.predict(x)
(y == preds).mean()


# investigate some of our options
## keyword arguments: look at the objects's effects in detail
### max_depth: determine how many branches are allowed
n_features = 200
x,y = datasets.make_classification(750,n_features,
                                   n_informative = 5)
training = np.random.choice([True,False], p=[.75,.25],
                            size = len(y))
accuracies = []
for md in np.arange(1,n_features+1):
    dt = DecisionTreeClassifier(max_depth = md)
    dt.fit(x[training], y[training])
    preds = dt.predict(x[~training])
    accuracies.append((preds == y[~training]).mean())

### plot    
fig,ax = plt.subplots(figsize = (7,5))
ax.plot(range(1,n_features+1),accuracies,color='k')
ax.set_title('Decision Tree Accuracy')
ax.set_ylabel('% Correct')
ax.set_xlabel('Max Depth')

### plot
N = 15
fig,ax = plt.subplots(figsize = (7,5))
ax.plot(range(1,n_features+1)[:N],accuracies[:N],color='k')
ax.set_ylabel('% Correct')
ax.set_xlabel('Max Depth')

### compute_importance: 
x,y = datasets.make_classification(750,200,n_informative = 5)
training = np.random.choice([True,False], p=[.75,.25],
                            size = len(y))
dt = DecisionTreeClassifier()                      
dt.fit(x,y)

### plot the importances
ne0 = dt.feature_importances_ != 0
y_comp = dt.feature_importances_[ne0]
x_comp = np.arange(len(dt.feature_importances_))[ne0]

fig,ax = plt.subplots(figsize = (7,5))
ax.bar(x_comp,y_comp)

#%% turning a decision tree model
x,y = datasets.make_classification(1000,20,
                                   n_informative = 3)
dt = DecisionTreeClassifier()
dt.fit(x,y)

# view a basic classifier fit
str_buffer = StringIO()
tree.export_graphviz(dt,out_file = str_buffer)
graph = pydot.graph_from_dot_data(str_buffer.getvalue())
graph[0].write('myfile.jpg')

## reduce the max depth value
dt = DecisionTreeClassifier(max_depth = 5).fit(x,y)

def plot_dt(model,filename):
    str_buffer = StringIO()    
    tree.export_graphviz(dt,out_file = str_buffer)
    graph = pydot.graph_from_dot_data(str_buffer.getvalue())
    graph[0].write(filename)    

plot_dt(dt,'myfile.jpg')

# Let's look at what happens when we use entropy as the splitting criteria:
dt = DecisionTreeClassifier(criterion='entropy',max_depth=5).fit(x, y)
plot_dt(dt, "entropy.png")
#%% http://scikit-learn.org/stable/modules/tree.html#classification

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
preds = clf.predict(iris.data)
(iris.target == preds).mean()

# export the tree in Graphviz format using the export_graphviz exporter
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

# use Graphviz’s dot tool to create a PDF file (or any other supported file type): 
import os
os.unlink('iris.dot')

# generate a PDF file (or any other supported file type) directly in Python with module pydotplus
import pydotplus
dot_data = tree.export_graphviz(clf,out_file='tree.dot') 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph[0].write_pdf("iris.pdf") 



from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  






































