'''
This package is for EEG data analysis with machine learning method.


This package contains six sub-packages:

"eeg_prepro" -- import raw eeg data and denoise 

"eeg_feature" -- extract eeg feature, include: 
       power spectral density, power percent,
       wavelet entropy, 
       AR coefficient

"feature_prepro" -- feature engineering, process include:
       1 feature_integre  2 feature_clean
       3 feature_reduce   4 feature_transform

"ml" -- machine learning method, include:
       1 SVM            2 Naive Bayes    3 Decision Tree
       4 BP             5 KNN            6 Logistic 
       7 random Tree    8 adaboosting    9 boosted trees
       10 bagged trees

"chart" -- Visualization the result, include:
       1 

"table" -- export some table, include
       1 

'''

# 1 load module
#import numpy as np 
#import pandas as pd 
#import matplotlib.pyplot as plt  

# 2 exception handling



# 3 __all__ property
__all__ = ['eeg_prepro',"eeg_feature", "feature_processing", "ml",
"chart", "table", "statis", "folder_file"]