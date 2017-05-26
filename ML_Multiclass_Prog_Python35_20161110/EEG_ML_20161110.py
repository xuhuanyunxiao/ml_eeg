# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:58:19 2016

@author: xuhuan
"""

## Improved：
#
#
#
#

## Improving：
# 
#
#
#
#

#%%

if 'ML' in dir():
    del ML

global ML
ML = {'FolderName':{},\
      'FileName':{},\
      'Experiment':{},\
      'Acquisition':{},\
      'Parameters':{'ImportRawData':{},\
                       'RawPreprocessing':{},\
                       'AfterRestrictAmplitude':{},\
                       'RawFeature':{},\
                       'PreprocessedFeature':{},\
                       'MachineLearning':{}
                       },\
      'Visualization':{'ImportRawData':{},\
                       'RawPreprocessing':{},\
                       'AfterRestrictAmplitude':{},\
                       'RawFeature':{},\
                       'PreprocessedFeature':{},\
                       'MachineLearning':{}
                       },\
      'History':{}}

#%% parameter setup
## Data Folder
### singal channel with 5s data
ML['FolderName']['DataFolder'] = r'D:\summerEEGdata\Data_For_Python'
                                            
## experimental introduction
ML['Experiment']['ExperimentName'] ='两种气味实验(香油、醋味、清醒)' 
ML['Experiment']['ExperimentType'] = 2
ML['Experiment']['ConditionName'] = ['perfume','vinegar','wake'] 
ML['Experiment']['ConditionNum'] = len(ML['Experiment']['ConditionName'])
ML['Experiment']['DayName'] = ['Day20160802','Day20160803','Day20160804',\
    'Day20160805','Day20160808','Day20160809','Day20160810','Day20160811',\
    'Day20160812','Day20160815','Day20160816','Day20160817','Day20160818',\
    'Day20160819','Day20160820','Day20160821','Day20160822','Day20160823',\
    'Day20160824','Day20160825','Day20160826','Day20160830','Day20160831',\
    'Day20160901','Day20160902','Day20160905','Day20160906','Day20160907',\
    'Day20160908','Day20160909','Day20160912','Day20160913','Day20160914',\
    'Day20160918','Day20160919','Day20160920','Day20160921','Day20160922',\
    'Day20160923','Day20160926','Day20160927','Day20160928','Day20161011',\
    'Day20161012','Day20161013','Day20161014','Day20161017','Day20161018',\
    'Day20161019','Day20161020','Day20161021','Day20161024','Day20161026',\
    'Day20161027','Day20161028']

## acquisition modality information
ML['Acquisition']['ChannelNum'] = 1
ML['Acquisition']['fs'] = 512

# 1 data loading
ML['Parameters']['ImportRawData']['IsImportRawData'] = 0

# 2 EEG data preprocessing
ML['Parameters']['RawPreprocessing']['IsRawPreprocessing'] = 0
ML['Parameters']['RawPreprocessing']['RawPreprocessingWay'] = 8
ML['Parameters']['RawPreprocessing']['locutoff'] = 1
ML['Parameters']['RawPreprocessing']['hicutoff'] = 20
ML['Parameters']['RawPreprocessing']['Amplitude'] = 150

# 3 calculate feature 
ML['Parameters']['RawFeature']['IsCalculateFeature'] = 1
ML['Parameters']['RawFeature']['FeatureType'] = 1
ML['Parameters']['RawFeature']['FeatureTypeName'] = ['PSD','PowerPecrcent','TimeSeries','FeatureCombine']

# 4 feature data preprocessing
ML['Parameters']['PreprocessedFeature']['IsFeaturePreprocessing'] = 0
ML['Parameters']['PreprocessedFeature']['FeaturePreprocessingWay'] = 1

# 5 machine learning
ML['Parameters']['MachineLearning']['IsMachineLearning'] = 0
ML['Parameters']['MachineLearning']['MachineLearningMethod'] = 1
ML['Parameters']['MachineLearning']['MachineLearningMethodName'] = ['SVM','BP','DecisionTree','NaiveBayes','KNN']

# 6 Visualization
ML['Visualization']['ImportRawData']['IsVisualImportRawData'] = 0
ML['Visualization']['RawPreprocessing']['IsVisualPreprocessedRawData'] = 0
ML['Visualization']['AfterRestrictAmplitude']['IsVisualAfterRestrictAmplitudeData'] = 0
ML['Visualization']['RawFeature']['IsVisualFeatureData'] = 1
ML['Visualization']['PreprocessedFeature']['IsVisualPreprocessedFeatureData'] = 0
ML['Visualization']['MachineLearning']['IsVisualMachineLearningResult'] = 0

#%% initialize parameter
# load useful function and library
from time import time

#import scipy as sp
#    import scipy.io
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as mtick

import sys
sys.path.append(r'D:\XH\analysis_prog\ML_Multiclass_Prog_Python35_20161110')

import py35_ML_Folder_and_File_Structure
py35_ML_Folder_and_File_Structure.py35_ML_Folder_and_File_Structure(ML)

#%% 1 import raw data

#%% 2 preprocessing EEG data

#%% 3 read feature data and label

# get proper feature data   
def FeatureStatistics(FeatureData,FeatureLabel):
    global ML
    count = []
    RowName = ['Condition_1','Condition_2','Condition_3']
    ColName = np.arange(len(FeatureData.ix[1,:]))
    mean = median = mode = Range = SD = SE = CV = skewness = kurtosis = \
    pd.DataFrame(index = RowName,columns = ColName)
    for Cond in np.arange(1,ML['Acquisition']['ChannelNum'] + 1):
        ConditionData = FeatureData.ix[FeatureLabel.ix[:,0]==Cond,:]
        Describe = ConditionData.describe() # [0 1 2 3 7] count mean std min max
        count.append(Describe.ix[0,0]);
        # central tendency: mean median mode
        mean.ix['Condition_' + str(Cond)] = Describe.ix[1].T
        median.ix['Condition_' + str(Cond)] = ConditionData.median().T
        mode.ix['Condition_' + str(Cond)] = ConditionData.mode().ix[0]
        # dispersion tendency: range SD SE CV
        Range.ix['Condition_' + str(Cond)] = Describe.ix[7].T - Describe.ix[3].T
        SD.ix['Condition_' + str(Cond)]  = Describe.ix[2].T
        SE.ix['Condition_' + str(Cond)] = SD.ix['Condition_' + str(Cond)]/np.sqrt(count[Cond - 1])
        CV.ix['Condition_' + str(Cond)] = SD.ix['Condition_' + str(Cond)]/mean.ix['Condition_' + str(Cond)]
        # distributive measurement: skewness kurtosis
        skewness.ix['Condition_' + str(Cond)] = ConditionData.skew().T
        kurtosis.ix['Condition_' + str(Cond)] = ConditionData.kurt().T
                    
    StaticsResult = {'count':count,'mean':mean,'median':median,'mode':mode, \
    'Range':Range,'SD':SD,'SE':SE,'CV':CV,'skewness':skewness,'kurtosis':kurtosis}
    # StaticsResult['mean'].ix['Condition_1']
    return StaticsResult

# plot feature's statistics result        
def VisualFeatureData(data):
    global ML

    
    params={
    'axes.labelsize': '25',       
    'xtick.labelsize':'10',
    'ytick.labelsize':'15',
    'lines.linewidth':2 ,
    'legend.fontsize': '15',
    'figure.figsize'   : '160, 90'    # set figure size
    }
    pylab.rcParams.update(params)            #set figure parameter
    #line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #set line style
    
    if ML['Visualization']['RawFeature']['Feature_meshon']:
        pass

    if ML['Visualization']['RawFeature']['Feature_boxon']:
        plt.boxplot()
        pass    

     
    if ML['Visualization']['RawFeature']['Feature_MSEon']:
        mean = FeatureStaticsResult['mean']
        SE = FeatureStaticsResult['SE']
        plt.figure(figsize = (8,4))
        for Cond in np.arange(1,ML['Acquisition']['ChannelNum'] + 1):                       
            x = np.arange(1,len(mean.ix['Condition_' + str(Cond)])+1)
            y = mean.ix['Condition_' + str(Cond)]
            y1 = mean.ix['Condition_' + str(Cond)] + SE.ix['Condition_' + str(Cond)]*3
            y2 = mean.ix['Condition_' + str(Cond)] - SE.ix['Condition_' + str(Cond)]*3
            plt.plot(x,y,'bo-',label='mean',markersize=10) 
            # in 'bo-', b is blue, o is O marker, - is solid line and so on)
            plt.plot(x,y1,'ro--',label='SE',markersize=10)
            plt.plot(x,y2,'ro--',label='SE',markersize=10)
            
        # title    
        plt.title('M + SE') 
        # axis:xlabel  xlim  xtick  xtickname
        plt.xlabel('power')
        plt.ylabel('frequecy Hz')
        plt.xlim([0,len(mean.ix['Condition_' + str(Cond)])+1])
        tick = range(1,len(mean.ix['Condition_' + str(Cond)])+1)
        ticklabel = []
        for i in tick:ticklabel.append(['Fea_' + str(i)])        
        plt.xticks(tick,ticklabel)
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt) 
        # grid
        axes = plt.gca() # get current axes
        axes.grid(True)  # add grid
        axes.yaxis.set_major_formatter(xticks) # set % format to ystick.
        # legend
        plt.legend(loc="upper right")  #set legend location
        # text
        # plt.text(s = '1',fontsize = 40)
        
        
        plt.show()
#        plt.savefig('D:\\commonNeighbors_CDF_snapshots.eps',dpi = 1000,bbox_inches='tight')

            

    if ML['Visualization']['RawFeature']['Feature_OtherFeatureStatistics']:
        pass
    
    
    return 'Feature had been plotted'
    
#
# load data
t0 = time()
if ML['Parameters']['RawFeature']['IsCalculateFeature']:
    if 'FeatureData' not in vars():
        FeatureData = pd.read_excel(ML['FileName']['FeatureType'] + '_FeatureData.xlsx',sheetname=3, header=None)
        FeatureLabel = pd.read_excel(ML['FileName']['FeatureType'] + '_FeatureLabel.xlsx',sheetname=3, header=None)
print("Time for loading data is %0.3fs" % (time() - t0))
        
t0 = time()        
if ML['Visualization']['RawFeature']['IsVisualFeatureData']:
    ML['Visualization']['RawFeature']['Feature_meshon'] = 0
    ML['Visualization']['RawFeature']['Feature_boxon'] = 0
    ML['Visualization']['RawFeature']['Feature_MSEon'] = 1
    ML['Visualization']['RawFeature']['Feature_OtherFeatureStatistics'] = 0  
    
    FeatureStaticsResult = FeatureStatistics(FeatureData,FeatureLabel);
    VisualFeatureData(FeatureStaticsResult)        
print("Time for plotting is %0.3fs" % (time() - t0))  
        

#%% 4 data wrangling
## 4.1 feature integration for multi-channel

## 4.2 feature clean: 


## 4.3 feature reduction: choosing feature with statistics


## 4.4 feature transformation: [0 1]





#%% 5 machine learning

if ML['Parameters']['MachineLearning']['IsMachineLearning']:
    MLdata = FeatureData
    MLlabel = FeatureLabel
    
    MLmethod = ML['Parameters']['MachineLearning']['MachineLearningMethod']
    if MLmethod == 1: # svm
        import py35_ML_SVM
        py35_ML_SVM.py35_ML_SVM(MLdata,MLlabel)
    elif MLmethod == 2: # decision tree
        pass
    elif MLmethod == 3: # bayes
        pass
    elif MLmethod == 4: # knn
        pass
    elif MLmethod == 5:
        pass
    elif MLmethod == 6:
        pass
    else:
        pass
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


