# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:11:21 2017

@author: yishikeji-01
"""
#%%
from ml_eeg.ml.ml_method import get_ml

import itertools
import pickle
from cycler import cycler

import numpy as np
import pandas as pd
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#%%
pylab.rcdefaults()
params={'axes.titlesize': 50,
        'axes.labelsize': 40,
        'axes.grid'    : 'True',
        'axes.prop_cycle': cycler('color',['cyan', 'indigo', 'seagreen', 
                            'yellow', 'blue','navy','turquoise', 'darkorange', 
                            'cornflowerblue', 'teal']),                    

        'xtick.labelsize':20,
        'ytick.labelsize':20,
        
        'lines.linewidth' :2,
        'lines.markersize':5,
        
        'legend.fontsize':30,
        
        'image.cmap':'Blues',
        'image.interpolation':'nearest',
   
        'figure.figsize' : '40, 30'    # set figure size
    }
pylab.rcParams.update(params)    

# plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
#                            cycler('linestyle', ['-', '--', ':', '-.'])))
#%%
def plot_precision_recall(precisions,recalls,average_precisions,
                          ConditionName,file_path):
    n_classes = len(ConditionName)

    # Plot Precision-Recall curve for each class
    plt.figure()
    plt.plot(recalls["micro"], precisions["micro"], color='gold',
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precisions["micro"]))
    for i in range(n_classes):
        plt.plot(recalls[i], precisions[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(ConditionName[i], average_precisions[i]))
    plt.plot([0, 1], [1, 0], color='cornflowerblue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve',loc='center')
    plt.legend(loc="lower right")
    plt.show()
    
    # save
    filename_fig = file_path + '_1_PR.jpg'
    savefig(filename_fig)

#%%
def plot_ROC_AUC(fpr,tpr,roc_auc,ConditionName,file_path):
    n_classes = len(ConditionName)
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], color='gold',
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label='ROC-AUC curve of class {0} (area = {1:0.2f})'
                       ''.format(ConditionName[i], roc_auc[i]))        
    plt.plot([0, 1], [0, 1], color='cornflowerblue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC curve',loc='center')
    plt.legend(loc="lower right")
    plt.show()
    
    # save
    filename_fig = file_path + '_2_ROC.jpg'
    savefig(filename_fig)

#%%
def table_confusion_matrix(cm,ticklabel,file_path,normalized = False,
                          title = 'Confusion matrix' ):
    plt.figure()
    plt.axes([0.1,0.2,1,0.75]) # [left, bottom, width, height]
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(ticklabel))
    plt.xticks(tick_marks, ticklabel, rotation=45)
    
    if normalized:
        plt.yticks(tick_marks[:4], ticklabel[:3] + ['average'])
    else:
        plt.yticks(tick_marks, ticklabel)
    
    disp_values = np.array(cm)
    thresh = disp_values.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, disp_values[i, j],
                 horizontalalignment="center",
                 fontsize=30,
                 color="white" if disp_values[i, j] > thresh else "black")

  #  plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()    
    
    # save
    filename_fig = file_path + '_3_' + title + '.jpg'
    savefig(filename_fig)
    
#%%
def evaluate_model(test_target,predict_target,predict_score,
                   ConditionName,file_path):   
    # statictis --------------------------------------------------------------
    accuracy = accuracy_score(test_target, predict_target)
    precision = precision_score(test_target, predict_target, average=None)
    recall = recall_score(test_target, predict_target, average=None)
    f1 = f1_score(test_target, predict_target, average=None)
    
    # Binarize the output
    test_bin_target = label_binarize(test_target,classes = np.arange(1,1 + len(ConditionName)))
    n_classes = test_bin_target.shape[1]
 
    # Precision-Recall -------------------------------------------------------
    # Compute Precision-Recall and plot curve
    precisions = dict()
    recalls = dict()
    average_precisions = dict()
    for i in range(n_classes):
        precisions[i], recalls[i], _ = precision_recall_curve(test_bin_target[:, i],
                                                              predict_score[:, i])
        average_precisions[i] = average_precision_score(test_bin_target[:, i], 
                                                        predict_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precisions["micro"], recalls["micro"], _ = precision_recall_curve(test_bin_target.ravel(),
                                                                      predict_score.ravel())
    average_precisions["micro"] = average_precision_score(test_bin_target, predict_score,
                                                          average="micro")   
    
    # ROC-AUC ----------------------------------------------------------------
    # Compute ROC curve and ROC area for each class
    fpr = dict() # false positive rates
    tpr = dict() # true positive rates
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_bin_target[:, i], predict_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_bin_target.ravel(), predict_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # confusion matrix --------------------------------------------------------
    # non-normalized confusion matrix
    cnf_matrix = confusion_matrix(test_target, predict_target)
    cnf = pd.DataFrame(cnf_matrix,
                       index = ConditionName,
                       columns = ConditionName)
    cnf['true_sum'] = cnf.sum(axis = 1)
    cnf.loc['pred_sum'] = cnf.sum(axis = 0)
    cnf['true_sum']['pred_sum'] = cnf['true_sum'].sum(axis = 0)/2
    
    # normalized confusion matrix
    cnf_norm_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_norm = pd.DataFrame(np.concatenate((cnf_norm_matrix , 
                                            np.array([cnf_norm_matrix.sum(axis = 1),
                                                      precision,recall,f1]).T),axis=1),
                            index = ConditionName,
                            columns = ConditionName + ['ACC','precision','recall','f1'])
    cnf_norm.loc['average'] = [0,0,0,accuracy,cnf_norm['precision'].mean(),
                               cnf_norm['recall'].mean(),
                               cnf_norm['f1'].mean()]
    cnf_norm['auc'] = list(average_precisions.values())
    
    # plot and table ----------------------------------------------------------
    plot_precision_recall(precisions,recalls,average_precisions,
                          ConditionName,file_path)
    plot_ROC_AUC(fpr,tpr,roc_auc,ConditionName,file_path)
    
    ticklabel=ConditionName +['sum']
    table_confusion_matrix(cnf,ticklabel,file_path)
    ticklabel=ConditionName +['ACC','precision','recall','f1','auc']
    table_confusion_matrix(round(cnf_norm,3),ticklabel,file_path,normalized = True,
                      title='normalized Confusion matrix')

    return cnf,cnf_norm
       
#%%
def ml_process(ML):
    # read data
    processed_featrue_set = pd.read_csv(ML['Feature_Process_File'] + '_sel_fea.csv')
    
    # split file
    target = processed_featrue_set['Condition']
    data = processed_featrue_set.ix[:,1:]

    train_set, test_set, train_target, test_target = \
        train_test_split(data, target, test_size=0.1, random_state=0)
    
    # machine-learning
    ml_method = ML['MLMethodName'][ML['MLMethod'] - 1]
    best_clf, predict_target,predict_score = get_ml(train_set,train_target,test_set,test_target,ml_method)
    
    # evaluation 
    ConditionName = ML['ConditionName']
    file_path = ML['Statis_ML_File']
    cnf,cnf_norm = evaluate_model(test_target,predict_target,predict_score,
                                  ConditionName,file_path)
        
    # update parameters
    new_param = {'cnf_matrix':cnf,
                 'cnf_norm_matrix':cnf_norm}
    ML.update(new_param)
    
    # save model and result    
    # 1 model
    filename_model = ML['ML_File'] + '_model.pkl'
    joblib.dump(best_clf, filename_model) 
    # clf = joblib.load(filename) 

    # 2 parameters
    filename_param = ML['ML_File'] + '_param.txt'
    pickle.dump(ML, open(filename_param, 'wb'))
    # ml_param = pickle.load(open(filename, 'rb'))
            
    # 3 result
    filename_result = ML['Statis_ML_File'] + '_result.xlsx'
    with pd.ExcelWriter(filename_result) as writer:
        cnf.to_excel(writer,sheet_name='cnf',startrow=0, startcol=0) 
        cnf_norm.to_excel(writer,sheet_name='cnf',startrow=7, startcol=0)
        
        
        