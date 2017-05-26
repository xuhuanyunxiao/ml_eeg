# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:05:17 2016

@author: yishikeji-01
"""

def py35_ML_Folder_and_File_Structure(ML):
    
    #global ML
    #%%
    import os
    
    DataFolder = ML['FolderName']['DataFolder']
    FeatureType = ML['Parameters']['RawFeature']['FeatureType']
    FeatureTypeName = ML['Parameters']['RawFeature']['FeatureTypeName']
    MachineLearningMethod = ML['Parameters']['MachineLearning']['MachineLearningMethod']
    MachineLearningMethodName = ML['Parameters']['MachineLearning']['MachineLearningMethodName']
    
    #%%
    
    if not os.path.exists(DataFolder + '\Result_Data_for_' + \
                          str(len(ML['Experiment']['DayName'])) + '_Days'):
        os.mkdir(DataFolder + '\Result_Data_for_' + \
                          str(len(ML['Experiment']['DayName'])) + '_Days') 
    
    ML['FolderName']['ResultFolder'] = DataFolder + '\Result_Data_for_' + \
                          str(len(ML['Experiment']['DayName'])) + '_Days'
    AllResultFolder = ML['FolderName']['ResultFolder']
    
    #%% One: result folder name
    # 1 raw data (.txt file)
    if not os.path.exists(AllResultFolder + '\ML_1_ImportData'):
        os.mkdir(AllResultFolder + '\ML_1_ImportData') 
    ML['FolderName']['ImportRawDataFolder'] = AllResultFolder + '\ML_1_ImportData'
    
    # 2 preprocessing EEG data
    if not os.path.exists(AllResultFolder + '\ML_2_RawPreprocessingResult'):
        os.mkdir(AllResultFolder + '\ML_2_RawPreprocessingResult') 
    ML['FolderName']['RawPreprocessingResultFolder'] = AllResultFolder + '\ML_2_RawPreprocessingResult'
    
    # 3 calculate feature
    if not os.path.exists(AllResultFolder + '\ML_3_FeatureResult_' \
                          + FeatureTypeName[FeatureType - 1]):
        os.mkdir(AllResultFolder + '\ML_3_FeatureResult_' \
                 + FeatureTypeName[FeatureType - 1]) 
    ML['FolderName']['FeatureResultFolder'] = AllResultFolder + '\ML_3_FeatureResult_'\
        + FeatureTypeName[FeatureType - 1]
    
    # 4 preprocessing feature 
    if not os.path.exists(AllResultFolder + '\ML_4_FeaturePreprocessingResult_'\
                          + FeatureTypeName[FeatureType - 1]):
        os.mkdir(AllResultFolder + '\ML_4_FeaturePreprocessingResult_' \
                 + FeatureTypeName[FeatureType - 1]) 
    ML['FolderName']['FeaturePreprocessingResultFolder'] = AllResultFolder + \
        '\ML_4_FeaturePreprocessingResult_' + FeatureTypeName[FeatureType - 1]
    
    # 5 machine learning
    if not os.path.exists(AllResultFolder + '\ML_5_MachineLearningResult_' \
                          + MachineLearningMethodName[MachineLearningMethod - 1]):
        os.mkdir(AllResultFolder + '\ML_5_MachineLearningResult_' \
                 + MachineLearningMethodName[MachineLearningMethod - 1]) 
    ML['FolderName']['MachineLearningResultFolder'] = AllResultFolder + \
        '\ML_5_MachineLearningResult_' + MachineLearningMethodName[MachineLearningMethod - 1]
    
    #%% Two: visualization folder name
    # 6 raw data (.txt file)
    if not os.path.exists(AllResultFolder + '\ML_6_VisualImportData'):
        os.mkdir(AllResultFolder + '\ML_6_VisualImportData') 
    ML['FolderName']['VisualImportRawDataFolder'] = AllResultFolder + '\ML_6_VisualImportData'
    
    # 7 preprocessing EEG data
    if not os.path.exists(AllResultFolder + '\ML_7_VisualRawPreprocessing'):
        os.mkdir(AllResultFolder + '\ML_7_VisualRawPreprocessing') 
    ML['FolderName']['VisualRawPreprocessingFolder'] = AllResultFolder + '\ML_7_VisualRawPreprocessing'
    
    # 8 calculate feature
    if not os.path.exists(AllResultFolder + '\ML_8_VisualFeatureResult_'\
                          + FeatureTypeName[FeatureType - 1]):
        os.mkdir(AllResultFolder + '\ML_8_VisualFeatureResult_'\
                 + FeatureTypeName[FeatureType - 1]) 
    ML['FolderName']['VisualFeatureResultFolder'] = AllResultFolder + \
        '\ML_8_VisualFeatureResult_' + FeatureTypeName[FeatureType - 1]
    
    # 9 preprocessing feature 
    if not os.path.exists(AllResultFolder + '\ML_9_VisualFeaPreproResult_'\
                          + FeatureTypeName[FeatureType - 1]):
        os.mkdir(AllResultFolder + '\ML_9_VisualFeaPreproResult_'\
                 + FeatureTypeName[FeatureType - 1]) 
    ML['FolderName']['VisualFeaturePreprocessingResultFolder'] = AllResultFolder + \
        '\ML_9_VisualFeaPreproResult_' + FeatureTypeName[FeatureType - 1]
    
    # 10 machine learning
    if not os.path.exists(AllResultFolder + '\ML_10_VisualMachineLearningResult_'\
                          + MachineLearningMethodName[MachineLearningMethod - 1]):
        os.mkdir(AllResultFolder + '\ML_10_VisualMachineLearningResult_'\
                 + MachineLearningMethodName[MachineLearningMethod - 1]) 
    ML['FolderName']['VisualMachineLearningResultFolder'] = AllResultFolder + \
        '\ML_10_VisualMachineLearningResult_' + MachineLearningMethodName[MachineLearningMethod - 1]
    
    #%% Three: file name
    # 1 raw data (.txt file)
    ML['FileName']['ImportRawDataFile'] = ML['FolderName']['ImportRawDataFolder'] +\
        '\ML_1_ImportData_ExpType' + str(ML['Experiment']['ExperimentType']) + '_' +\
        str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData'                                  
    
    # 2 preprocessing EEG data
    ML['FileName']['RawPreprocessing'] = ML['FolderName']['RawPreprocessingResultFolder'] +\
        '\ML_2_RawPrepro_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    
    ML['FileName']['RawPreprocessingRestrictAmplitude'] = ML['FolderName']['RawPreprocessingResultFolder'] +\
        '\ML_2_RawPrepro_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    
    # 3 calculate feature
    ML['FileName']['FeatureType'] = ML['FolderName']['FeatureResultFolder'] +\
        '\ML_3_FeatureResult_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_FeaType' + str(ML['Parameters']['RawFeature']['FeatureType']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    ML['FileName']['FeatureStatictis'] = ML['FolderName']['VisualFeatureResultFolder'] +\
        '\ML_8_VisualFeature_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_FeaType' + str(ML['Parameters']['RawFeature']['FeatureType']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    # 4 preprocessing feature 
    ML['FileName']['FeaturePreprocessing'] = ML['FolderName']['FeaturePreprocessingResultFolder'] +\
        '\ML_4_FeaturePreprocessing_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_FeaType' + str(ML['Parameters']['RawFeature']['FeatureType']) +\
        '_FeaPreproWay' + str(ML['Parameters']['PreprocessedFeature']['FeaturePreprocessingWay']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    ML['FileName']['PreprocessedFeatureStatictis'] = ML['FolderName']['VisualFeaturePreprocessingResultFolder'] +\
        '\ML_9_VisualFeaturePreprocessingResult_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_FeaType' + str(ML['Parameters']['RawFeature']['FeatureType']) +\
        '_FeaPreproWay' + str(ML['Parameters']['PreprocessedFeature']['FeaturePreprocessingWay']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    # 5 machine learning
    ML['FileName']['MachineLearning'] = ML['FolderName']['MachineLearningResultFolder'] +\
        '\ML_5_MachineLearning_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_FeaType' + str(ML['Parameters']['RawFeature']['FeatureType']) +\
        '_FeaPreproWay' + str(ML['Parameters']['PreprocessedFeature']['FeaturePreprocessingWay']) +\
        '_MLmethod' + str(ML['Parameters']['MachineLearning']['MachineLearningMethod']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
    
    ML['FileName']['MachineLearningStatictis'] = ML['FolderName']['VisualMachineLearningResultFolder'] +\
        '\ML_10_VisualMachineLearningResult_ExpType' + str(ML['Experiment']['ExperimentType']) +\
        '_RawPreproWay' + str(ML['Parameters']['RawPreprocessing']['RawPreprocessingWay']) +\
        '_FeaType' + str(ML['Parameters']['RawFeature']['FeatureType']) +\
        '_FeaPreproWay' + str(ML['Parameters']['PreprocessedFeature']['FeaturePreprocessingWay']) +\
        '_MLmethod' + str(ML['Parameters']['MachineLearning']['MachineLearningMethod']) +\
        '_Amplitude' + str(ML['Parameters']['RawPreprocessing']['Amplitude']) +\
        '_' + str(ML['Acquisition']['ChannelNum']) + 'Channel_' +\
        str(len(ML['Experiment']['DayName'])) + 'DaysData' 
       
                
    return

if __name__ == '__main__':
     
    py35_ML_Folder_and_File_Structure()
