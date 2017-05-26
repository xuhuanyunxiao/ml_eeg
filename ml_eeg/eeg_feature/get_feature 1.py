# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:50:19 2017

@author: yishikeji-01
"""

#%%
import pandas as pd
import numpy as np
import os


#%% ################### 1
def psd_mne(ML,eeg_data_label):
    '''
    
    '''

    from mne.time_frequency import psd_welch  
    from ml_eeg.eeg_prepro import mne_eeg
    
    fs = ML['fs']
    raw_data, raw_label = mne_eeg.creat_mne_object(eeg_data_label,fs)
    
    # label
    psd_label = pd.DataFrame(np.array(raw_label),columns = ['Condition'])  
    
    # psd data
    fmin = ML['locutoff']
    fmax = ML['hicutoff']
    n_fft = fs / ML['f_resolution']
    psd, freqs = psd_welch(raw_data, \
                           fmin=fmin, fmax=fmax, n_fft=n_fft, n_overlap=25, picks=None)
    
    psd_freqs = {'psd_freqs':freqs};
    ML.update(psd_freqs) # 更新 ML  
    
    def combine_data_label(data,label,freqs):
        data = pd.DataFrame(psd)
        # label = psd_label
        column = []
        for i in freqs: # max(freqs) < 100
            if i < 10:
                column.append('psd_0' + str(i))
            else:
                column.append('psd_' + str(i))
        data.columns = list(column)
        feature_data_label = label.join(data,how='outer')
        return feature_data_label
        
    # data and label
    feature_data_label = combine_data_label(psd,psd_label,freqs)    
    
    if ML['PSD_log']: # 后面特征无量纲化时有对数转换，此处不需要了
        psds = 20 * np.log10(psd)  # scale to dB
        feature_data_label = combine_data_label(psds,psd_label,freqs)
            
    return feature_data_label

#%% ################### 2
def psd_percent(ML,eeg_data_label):
    '''
    A:1-3Hz B:4-7Hz C:8-13Hz D1:14-20 D2:21-30 E1:31-50 E2:51-96
    δ（德尔塔）θ（西塔）α( 阿而法)β( 贝塔)γ(伽马）
    '''
    feature_data = psd_mne(ML,eeg_data_label)
    data_sum = feature_data.ix[:,1:].sum(axis = 1)
                    
    if ML['locutoff'] == 1:
        Del_name ='Delta_1_3'
        Del_num = np.arange(1,4)
    elif ML['locutoff'] == 3:
        Del_name ='Delta_3'
        Del_num = 3
    
    indexs = {'Delta':Del_num,'Theta':np.arange(4,8),'Alpha':np.arange(8,14),
              'Beta':np.arange(14,21),'Beta1':np.arange(14,21),'Beta2':np.arange(21,31),
              'Gamma':np.arange(31,41),'Gamma1':np.arange(31,51),'Gamma2':np.arange(51,97)}

    if ML['hicutoff'] == 20:        
        index = [indexs['Delta'],indexs['Theta'],indexs['Alpha'],indexs['Beta']]
        columns = [Del_name,'Theta_4_7','Alpha_8_13','Beta_14_20']
    if ML['hicutoff'] == 40:        
        index = [indexs['Delta'],indexs['Theta'],indexs['Alpha'],indexs['Beta1']
                 ,indexs['Beta2'],indexs['Gamma']]
        columns = [Del_name,'Theta_4_7','Alpha_8_13','Beta1_14_20'
                   ,'Beta2_21_30','Gamma_31_40']
    if ML['hicutoff'] == 96:        
        index = [indexs['Delta'],indexs['Theta'],indexs['Alpha'],indexs['Beta1']
                 ,indexs['Beta2'],indexs['Gamma1'],indexs['Gamma2']]
        columns = [Del_name,'Theta_4_7','Alpha_8_13','Beta1_14_20'
                   ,'Beta2_21_30','Gamma1_31_50','Gamma2_50_96']
                   
    def get_percent():
        for ind,item in enumerate(columns):
            if (ML['locutoff'] == 3) & (item == 'Delta_3'):
                percent_data[item] = feature_data.ix[:,index[ind]-2] / data_sum
            else:
                percent_data[item] = feature_data.ix[:,index[ind]-2].sum(axis = 1) / data_sum
        return percent_data
        
    percent_data = pd.DataFrame(columns = columns)
    percent_data = get_percent()
    feature_data_label = pd.DataFrame(feature_data['Condition']).join(percent_data,how='outer')
    
    return feature_data_label
    
#%% ################### 3
def time_featrue():
    '''
    
    '''
    pass

#%% ################### 4
def AR_feature():
    '''

    '''
    pass

#%% ################### 5
def combined_feature():
    '''
    
    '''
    pass

#%% ################### main part
def feature_set(ML):
    '''
    
    '''
      
    # read data
    eeg_data_label = pd.read_csv(ML['EEG_Import_File'] + '.csv')
    
    
    
    
    if ML['FeatureType'] == 1:
        feature_data_label = psd_mne(ML,eeg_data_label)
    elif ML['FeatureType'] == 2:
        feature_data_label = psd_percent(ML,eeg_data_label)
    elif ML['FeatureType'] == 3:
        feature_data_label = time_featrue(ML,eeg_data_label)
    elif ML['FeatureType'] == 4:
        feature_data_label = AR_feature(ML,eeg_data_label)
    elif ML['FeatureType'] == 5:
        feature_data_label = combined_feature(ML,eeg_data_label)
        
    # save 
    if not os.path.isfile(ML['Feature_File'] + '.csv'):
        feature_data_label.to_csv(ML['Feature_File'] + '.csv',index = False) 




