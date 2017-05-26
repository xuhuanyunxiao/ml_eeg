# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:53:47 2016

@author: yishikeji-01
"""

#%%
def denoise_eeg(ML):
    from ml_eeg.eeg_prepro import mne_eeg
    import pandas as pd
    import pickle
    
    # read data
    eeg_data_label = pd.read_csv(ML['EEG_Import_File'] + '.csv')
        
    fs = ML['fs']
    ChannelNum = ML['ChannelNum']
    ConditionName = ML['ConditionName']
    clearn_data,clean_label = mne_eeg.creat_mne_epoch_object(eeg_data_label,ConditionName,
                           fs = fs,ChannelNum = ChannelNum,
                           baseline=(None, None),detrend  = 0)
    
    # Apply the signal space projection (SSP) operators to the data.
    # epoch_data = epoch_data.apply_proj()    
    
    # IIR
    #Fp1 = ML['locutoff']
    #Fp2 = ML['hicutoff']
    #denoise_data = mne.filter.band_pass_filter(epoch_data.get_data(), fs, Fp1, Fp2)
    #denoise_label = epoch_label
    
    # save 
    filename_data = ML['EEG_Prepro_File'] + '_clearn_data.txt'
    pickle.dump(clearn_data, open(filename_data, 'wb'))
    filename_label = ML['EEG_Prepro_File'] + '_clearn_label.txt'
    pickle.dump(clean_label, open(filename_label, 'wb'))
    # ml_param = pickle.load(open(filename, 'rb'))
        
    return clearn_data,clean_label
