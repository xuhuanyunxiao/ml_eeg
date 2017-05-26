# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:01:34 2016

@author: yishikeji-01
"""

#%%
import numpy as np    
import mne
    
#%%
def creat_mne_raw_object(eeg_data_label,fs):
    
    # 1 load data
#    eeg_data_label = pd.read_csv(ML['ImportRawDataFile'] + '.csv')
    
    # 2 change the shape of data
    data = []
    for i in range(len(eeg_data_label)):
        data.append(eeg_data_label.ix[i,list(np.arange(3,eeg_data_label.shape[1]))])
        
    # Numpy array of size sample_N X time_N
    data = np.array(data)
    
    # Definition of channel types and names.
    ch_types = ['eeg'] * len(eeg_data_label)
    ch_names = ['Chan1_S' + str(i) for i in range(1, len(eeg_data_label) + 1)]
    
    # 3 Creation of info dictionary
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    
    raw_data = mne.io.RawArray(data, info)
    raw_label = list(eeg_data_label['0'])

    return raw_data, raw_label

#%%

def creat_mne_epocharray_object(eeg_data_label,fs = 512,ChannelNum = 1):
    
    raw_label = [int(i) for i in eeg_data_label['0']]
    data = np.array(eeg_data_label.ix[:,3:])   
    
    # 3D array of shape (n_epochs, n_channels, n_samples)
    raw_data = np.zeros([data.shape[0],ChannelNum,fs])
    for i in range(data.shape[0]):
        for j in range(ChannelNum):
            raw_data[i,j] = data[i,fs*j+0:fs*j+512]    
    
    # Definition of channel types and names.
    ch_types = ['eeg'] * ChannelNum
    ch_names = ['channel_' + str(i) for i in range(1, ChannelNum + 1)]
    
    # 3 Creation of info dictionary
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    
    
    raw_data = mne.EpochsArray(raw_data, info)
    # mne.EpochsArray(data, info, events=None, tmin=0, event_id=None, 
    #              reject=None, flat=None, reject_tmin=None, reject_tmax=None, 
    #              baseline=None, proj=True, verbose=None)
    
    return raw_data, raw_label
 
#%%

def creat_mne_epoch_object(eeg_data_label,ConditionName,
                           fs = 512,ChannelNum = 1,
                           baseline=(None, None),
                            detrend  = 0):
    
    # Creation of info dictionary
    ch_types = ['eeg'] * ChannelNum
    ch_names = ['channel_' + str(i) for i in range(1, ChannelNum + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    
    # data:array, shape (n_channels, n_times)
    raw_label = [int(i) for i in eeg_data_label['0']]
    epoch_label = raw_label
    eeg_data = np.array(eeg_data_label.ix[:,3:])
    
#    data = []
#    for d in range(eeg_data.shape[0]):
#        for x in list(eeg_data[d,:]):
#            data.append(x)
#    da = np.array(data)
#    data = da.reshape(1,da.shape[0])
    data =  eeg_data.reshape(1,eeg_data.size)   
    # example: 按行分割
    # a = np.arange(50),b = a.reshape(5,10),c = b.reshape(1,50)
    raw_data = mne.io.RawArray(data, info)

    # events：array, shape (n_events, 3)
    # The first column corresponds to sample number. 
    # The second column is reserved for the old value of the trigger channel 
    # at the time of transition, but is currently not in use. 
    # The third column is the trigger id (amplitude of the pulse).
    events = []
    for x,y in enumerate(raw_label):
        events.append([0+x*fs,0,y])
    events = np.array(events)
    
    event_id = {}
    for x,y in enumerate(ConditionName): event_id[y] = x +1

    # detrend:0 is a constant (DC) detrend, 1 is a linear detrend.    
    # baseline:If baseline is equal to (None, None) all the time interval is used
    epoch_data = mne.Epochs(raw_data, events,event_id = event_id,
                            tmin = 0,tmax= 0.999,baseline=(None, None),
                            detrend  = 0,
                            proj=None,reject_by_annotation=None)
    
    # epoch_data.get_data(),return 3D array of shape (n_epochs, n_channels, n_samples)
    # epoch_data.resample(100) # 重采样到100Hz

    return epoch_data,epoch_label

