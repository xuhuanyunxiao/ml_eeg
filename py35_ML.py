# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:01:05 2016

changed in 
20170204

@author: xuhuan
"""

#%%
code_path = 'D:\XH\py35_code'
# 修改工作路径
import os
# os.getcwd()
os.chdir(code_path)

# 增加搜索路径
import sys 
sys.path.append(code_path)

#%% 建立变量
if 'ML' in dir():
    del ML
    
ML = {}

#%% 设定参数
# 原始数据
ML['DataFolder'] = r'D:\summerEEGdata\exercise_data'
ML['TimeRange'] = 1 # unit is senond（s) 

## experimental introduction
ML['ExperimentName'] ='两种气味实验(香油、醋味、清醒)' 
ML['ExperimentType'] = 1
ML['ConditionName'] = ['perfume','vinegar','wake'] 
ML['ConditionNum'] = len(ML['ConditionName'])
ML['DayName'] = ['Day20161114','Day20161115','Day20161116']

## acquisition modality information
ML['ChannelNum'] = 1
ML['fs'] = 512

# 1 data loading
ML['IsImportEEG'] = 1

# 2 EEG data preprocessing
ML['IsEEGPrepro'] = 1
ML['EEGPreproWay'] = 8
ML['locutoff'] = 3  # 为 1 或者 3
ML['hicutoff'] = 96 # 为 20 、 40 或者 96
ML['f_resolution'] = 1  # 频率分辨率 = 1/t_window
ML['Amplitude'] = 150

# 3 calculate feature 
ML['IsCalculateFeature'] = 1
ML['FeatureType'] = 1
ML['FeatureTypeName'] = ['PSD','PowerPecrcent','TimeSeries','FeatureCombine']
ML['PSD_log'] = 0 # 是否将 psd值 进行对数转换。特征无量纲化时有对数转换，此处不需要了

# 4 feature data processing
ML['IsFeatureProcess'] = 1
ML['FeatureProcessWay'] = 1

# 5 machine learning
ML['IsML'] = 0
ML['MLMethod'] = 1
ML['MLMethodName'] = ['SVM','BP','DecisionTree','NaiveBayes','KNN','Logistic',
                      'RandomTree','GDBT','Adaboosting','BaggedTrees','Voting']

# 6 Visualization
ML['IsVisualImportedEEG'] = 1
ML['IsVisualPreprocessedEEG'] = 1
#ML['IsVisualAfterRestAmplEEG'] = 0
ML['IsVisualFeature'] = 1
ML['IsVisualPreprocessedFeature'] = 0
#ML['IsVisualML'] = 0

#%% EEG数据处理
from ml_eeg.folder_file import folder_file_path
folder_file_path.folder_path_struction(ML)
folder_file_path.file_path_struction(ML)

#%% 1 导入EEG数据
if ML['IsImportEEG']:
    if not os.path.isfile(ML['EEG_Import_File'] + '.csv'):
        from ml_eeg.eeg_prepro import import_eeg
        import_eeg.load_data(ML)

if ML['IsVisualImportedEEG']:
    if not os.path.isfile(ML['Statis_EEG_Import_File'] + '.xlsx'):
        from ml_eeg.chart import visual_data
        visual_data.view_imported_eeg(ML,thresh = 200)
#%% 2 EEG预处理
from ml_eeg.eeg_prepro import processing_eeg
if not os.path.isfile(ML['EEG_Prepro_File'] + '_clearn_data.txt'):
    processing_eeg.denoise_eeg(ML)

if ML['IsVisualPreprocessedEEG']:
    if not os.path.isfile(ML['Statis_EEG_Prepro_File'] + '.xlsx'):
        from ml_eeg.chart import visual_data
        visual_data.view_processed_eeg(ML,thresh = 200)    
#%% 3 特征计算
if ML['IsCalculateFeature']:    
    if not os.path.isfile(ML['Feature_File'] + '_feature_data.csv'):
        from ml_eeg.eeg_feature import get_feature
        get_feature.feature_set(ML)
        
#%% 4 特征处理
if ML['IsFeatureProcess']:    
    from ml_eeg.feature_processing import feature_prepro,feature_engineering
    feature_prepro.clean_featrue_set(ML)
    feature_engineering.processed_featrue_set(ML)

#%% 5 机器学习
if ML['IsML']: 
    if not os.path.isfile(ML['Statis_ML_File'] + '_result.xlsx'):
        from ml_eeg.ml import ml_main
        ml_main.ml_process(ML)

#%%

from timeit import Timer

vocab_size = 10000
setup_list = 'import random;vocab = range(%d)' % vocab_size
setup_set = 'import random;vocab = set(range(%d))' % vocab_size

statement = 'random.randint(0,%d) in vacab' % vocab_size * 2

Timer(statement,setup_list).timeit(1000)






















