# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:21:10 2017

@author: yishikeji-01
"""


import pandas as pd
import os

#################### 1
def feature_integre():
    '''
    是否进行特征集成（不同数据源，如多通道数据）
    '''
    
    pass

#################### 3
def feature_clean():
    '''
    是否进行特征清洗（离群点、缺失值）
    以去样本为主
    '''
    
    pass


    
#################### main part
def clean_featrue_set(ML):
    '''
    
    '''
    # read data
    feature_data_label = pd.read_csv(ML['Feature_File'] + '.csv')
    
    clean_featrue_set = feature_data_label
    
    # save 
    if not os.path.isfile(ML['Feature_Process_File'] + '_clean_fea.csv'):
        clean_featrue_set.to_csv(ML['Feature_Process_File'] + '_clean_fea.csv',index = False) 


