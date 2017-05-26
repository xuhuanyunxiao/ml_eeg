# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:08:35 2016

@author: yishikeji-01
"""

def folder_path_struction(ML):
    '''
    Build a folder systme for intermediate files and result.
    '''
    import os 
    
    # 建立变量
    DataFolders = ML['DataFolder']
    FeatureType = ML['FeatureType']
    FeatureTypeName = ML['FeatureTypeName']
    MachineLearningMethod = ML['MLMethod']
    MachineLearningMethodName = ML['MLMethodName']
    
    DataFolder = DataFolders + '\Result_Data_for_' + str(len(ML['DayName'])) + '_Days'
    FolderStruct = {'ResultFolder':DataFolder,
                    'EEG_Import_Folder':DataFolder + '\ML_1_EEG_Import',
                    'EEG_Prepro_Folder':DataFolder + '\ML_2_EEG_Prepro',
                    'Feature_Folder':DataFolder + '\ML_3_Feature_'\
                        + FeatureTypeName[FeatureType - 1],
                    'Feature_Process_Folder':DataFolder + '\ML_4_Feature_Process_'\
                        + FeatureTypeName[FeatureType - 1],
                    'ML_Folder':DataFolder + '\ML_5_ML_'\
                        + MachineLearningMethodName[MachineLearningMethod - 1],
                    'Visual_EEG_Import_Folder':DataFolder + '\ML_6_Visual_EEG_Import',
                    'Visual_EEG_Prepro_Folder':DataFolder + '\ML_7_Visual_EEG_Prepro',
                    'Visual_Feature_Folder':DataFolder + '\ML_8_Visual_Feature_'\
                        + FeatureTypeName[FeatureType - 1],
                    'Visual_Feature_Process_Folder':DataFolder + '\ML_9_Visual_Feature_Process_'\
                        + FeatureTypeName[FeatureType - 1],
                    'Visual_ML_Folder':DataFolder + '\ML_10_Visual_ML_'\
                        + MachineLearningMethodName[MachineLearningMethod - 1]
                    }   
                    
    ML.update(FolderStruct) # 更新 ML 中 文件夹 的路径                    
    def make_folder(Folder_Name):
        if not os.path.exists(Folder_Name):
            os.mkdir(Folder_Name)     
    for Folder_Name in FolderStruct.values():make_folder(Folder_Name)

        
def file_path_struction(ML):
    '''
    Build a file systme 
    '''

    # 建立变量    
    file_name_tail = '_' + str(ML['ChannelNum']) + 'Channel_' +\
        str(len(ML['DayName'])) + 'DaysData'
    
    FileStruct = {'EEG_Import_File':ML['EEG_Import_Folder'] +\
        '\ML_1_EEG_Import_ExpType' + str(ML['ExperimentType']) +  file_name_tail,
                  'Statis_EEG_Import_File':ML['Visual_EEG_Import_Folder'] +\
        '\ML_6_EEG_Import_ExpType' + str(ML['ExperimentType']) +  file_name_tail,
                  'EEG_Prepro_File':ML['EEG_Prepro_Folder'] +\
        '\ML_2_EEG_Prepro_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) + file_name_tail,
                  'Statis_EEG_Prepro_File':ML['Visual_EEG_Prepro_Folder'] +\
        '\ML_7_EEG_Prepro_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) + file_name_tail,
                  'EEG_RestAmp_File':ML['EEG_Prepro_Folder'] +\
        '\ML_2_EEG_Prepro_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,   
                  'Statis_EEG_RestAmpl_File':ML['Visual_EEG_Prepro_Folder'] +\
        '\ML_7_Visual_EEG_Prepro_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,           
                  'Feature_File':ML['Feature_Folder'] +\
        '\ML_3_Feature_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_FeaType' + str(ML['FeatureType']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,
                  'Statis_Feature_File':ML['Visual_Feature_Folder'] +\
        '\ML_8_Visual_Feature_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_FeaType' + str(ML['FeatureType']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,
                  'Feature_Process_File':ML['Feature_Process_Folder'] +\
        '\ML_4_Feature_Process_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_FeaType' + str(ML['FeatureType']) +\
        '_FeaProcessWay' + str(ML['FeatureProcessWay']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,
                  'Statis_Feature_Process_File':ML['Visual_Feature_Process_Folder'] +\
        '\ML_9_Visual_Feature_Process_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_FeaType' + str(ML['FeatureType']) +\
        '_FeaProcessWay' + str(ML['FeatureProcessWay']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,
                  'ML_File':ML['ML_Folder'] +\
        '\ML_5_ML_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_FeaType' + str(ML['FeatureType']) +\
        '_FeaProcessWay' + str(ML['FeatureProcessWay']) +\
        '_MLmethod' + str(ML['MLMethod']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,
                  'Statis_ML_File':ML['Visual_ML_Folder'] +\
        '\ML_10_Visual_ML_ExpType' + str(ML['ExperimentType']) +\
        '_EEGPreproWay' + str(ML['EEGPreproWay']) +\
        '_FeaType' + str(ML['FeatureType']) +\
        '_FeaProcessWay' + str(ML['FeatureProcessWay']) +\
        '_MLmethod' + str(ML['MLMethod']) +\
        '_Amplitude' + str(ML['Amplitude']) +\
        file_name_tail,
                  }
                  
    ML.update(FileStruct) # 更新 ML 中 文件 的路径 
     
