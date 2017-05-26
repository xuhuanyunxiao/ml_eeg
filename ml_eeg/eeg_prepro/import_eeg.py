# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:31:54 2016

@author: yishikeji-01
"""

def load_data(ML):
    '''
    
    '''
    import os
    
    import numpy as np
    import pandas as pd
    
    # 参数设置
    ConditionName = ML['ConditionName']
    DayName = ML['DayName']
    DataFolder = ML['DataFolder']
    ChannelNum = ML['ChannelNum']   

    # 建立变量 shape (n_epochs, n_channels, n_times)
    data_column = np.arange(ChannelNum * ML['TimeRange'] * ML['fs'] + 3)
    eeg_data_label = pd.DataFrame(columns = list(data_column))
    
#    column = []
#    for i in range(1,1 + ChannelNum):column.append('Channel_' + str(i))
#    EEG_Data = pd.DataFrame(columns=column)
#    EEG_Label = pd.DataFrame(columns=('Condition', 'Day', 'day_file_N'))
    
    File_N = 0

# 只考虑单通道    
#    for Chan in range(1,ChannelNum + 1):
    for Cond in range(1,len(ConditionName)+1):
        for Day in range(1,len(DayName)+1):
            FilePath = DataFolder + '\\' + DayName[Day - 1] + '\\' + ConditionName[Cond - 1] 
            FileNames=os.listdir(FilePath) 
            os.chdir(FilePath)
            day_file_N = 0                
            if (len(FileNames) > 0): # 列表不为空
                for filename in FileNames:
                    File_N += 1
                    day_file_N += 1
                    eeg = open(filename,'r').readlines() # 读入数据
                    eegs =[num.strip().split('\t') for num in eeg]
                    data = []
                    for x in eegs[0]:data.append(int(x))
                    
                    eeg_data_label.loc[File_N] = [Cond, Day, day_file_N] + data
#                    EEG_Data.loc[File_N] = [data]
#                    EEG_Label.loc[File_N] = Cond, Day, day_file_N
#
#    # 合并标签与数据
#    eeg_data_label = pd.DataFrame(EEG_Label['Condition']).join(EEG_Data,how='outer')

    # save
    if not os.path.isfile(ML['EEG_Import_File'] + '.csv'):
        eeg_data_label.to_csv(ML['EEG_Import_File'] + '.csv',index = False)   

    
    return eeg_data_label


#%%
    # 建立变量 shape (n_epochs, n_channels, n_times)
#    eeg_data_list = [[[]]]
#    eeg_label_list = []
#    File_N = -1
#    
#    for Cond in range(1,len(ConditionName)+1):
#        for Day in range(1,len(DayName)+1):
#            FilePath = DataFolder + '\\' + DayName[Day - 1] + '\\' + ConditionName[Cond - 1] 
#            FileNames=os.listdir(FilePath) 
#            os.chdir(FilePath)            
#            day_file_N = 0                
#            if (len(FileNames) > 0): # 列表不为空
#                for filename in FileNames:
#                    day_file_N += 1
#                    
#                    eeg = open(filename,'r').readlines() # 读入数据
#                    data = [int(x) for x in [num.strip().split('\t') for num in eeg]]
#                    
#                    if 1: # 单通道 :channel 1
#                        eeg_data_list.append(data)
#                        eeg_label_list.append([Cond, Day, day_file_N])
#                        File_N += 1
#                    else: # channel2,channel3......
#                        eeg_data_list[File_N].append(data)
                        
                    
    
    
    
    
