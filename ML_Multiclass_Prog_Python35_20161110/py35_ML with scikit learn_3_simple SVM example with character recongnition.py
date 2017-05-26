# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:41:22 2016

@author: yishikeji-01
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma = 0.001,C = 100)

x,y = digits.data[:-1],digits.target[:-1]
clf.fit(x,y)

print('-----------------------------------------------------------')
print(clf.predict(digits.data[-2]))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()






















