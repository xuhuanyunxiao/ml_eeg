# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:10:41 2016

@author: yishikeji-01
"""

def featrue_staic(feature_data_label):
    # group
    group = feature_data_label.groupby(['Condition'])
    
    # 
    Men = group.mean()
    Med = group.median()
    SD = group.std()  
    Range = group.max() - group.min()
    Skew = group.skew()
    Corr = group.corr()


    


def plot_psd(feature_data_label):
    '''
    
    '''
    
    import matplotlib.pyplot as plt    
    import pandsa as pd
    
    featrue_staic(feature_data_label)
    
    for i in np.arange(1, 1 + len(feature_data_label['Condition'].unique())):
        
    

