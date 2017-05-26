# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:21:10 2017

@author: yishikeji-01
"""

import pandas as pd
import numpy as np
import os

#################### 1
def feature_construct():
    '''
    增加特征：构造新特征
    
    常见的数据变换有基于多项式的、基于指数函数的、基于对数函数的。
    '''
    
    pass
    
#    # 1 多项式转换
#    from sklearn.preprocessing import PolynomialFeatures
#    #参数degree为度，默认值为2
#    PolynomialFeatures().fit_transform(iris.data)
#    
#    
#    # 2 自定义转换函数为对数函数的数据变换
#    from numpy import log1p
#    from sklearn.preprocessing import FunctionTransformer
#    #第一个参数是单变元函数
#    FunctionTransformer(log1p).fit_transform(iris.data)

#################### 2
class FeatureSelection():
    '''
      ● Filter：过滤法，按照发散性或者相关性对各个特征进行评分，
              设定阈值或者待选择阈值的个数，选择特征。
      ● Wrapper：包装法，根据目标函数（通常是预测效果评分），
              每次选择若干特征，或者排除若干特征。
      ● Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，
              得到各个特征的权值系数，根据系数从大到小选择特征。
              类似于Filter方法，但是是通过训练来确定特征的优劣。
    '''
    
    def filter_method(data,target,method):
        '''
        1 方差选择法(var):  #参数threshold为方差的阈值
            此处用这种方法不合适：psd时，方差太大，7972-11306315；psd_percent时，方差太小，0.01
        2 相关系数法(cor)
        使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值。        
        3 卡方检验(chi)
        经典的卡方检验是检验定性自变量对定性因变量的相关性。        
        4 互信息法(infro)
        经典的互信息也是评价定性自变量对定性因变量的相关性的
        
        以上后三种方法也不合适：无法确定选几个特征才合适
        '''
        
        if method == 'var':
            from sklearn.feature_selection import VarianceThreshold    
            selected_featrue_set = VarianceThreshold(threshold=0).fit_transform(data)
        elif method == 'cor':
            from sklearn.feature_selection import SelectKBest
            from scipy.stats import pearsonr
            #选择K个最好的特征，返回选择特征后的数据
            #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
            #输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
            #在此定义为计算相关系数
            #参数k为选择的特征个数
            selected_featrue_set = SelectKBest(lambda X, Y:np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(data,target)
        elif method == 'chi':
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import chi2
            selected_featrue_set = SelectKBest(chi2, k=2).fit_transform(data,target)
        elif method == 'infro':       
            from sklearn.feature_selection import SelectKBest
            from minepy import MINE
            #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
            def mic(x, y):
                m = MINE()
                m.compute_score(x, y)
                return (m.mic(), 0.5)
            selected_featrue_set = SelectKBest(lambda X, Y:np.array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(data, target)    
            
        return selected_featrue_set
            
    def wrapper_method(data,target,method):
        '''
        递归特征消除法
        递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，
        再基于新的特征集进行下一轮训练。
        
        此方法也不合适：无法确定选几个特征才合适
        '''
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        #参数estimator为基模型
        #参数n_features_to_select为选择的特征个数
        selected_featrue_set = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(data,target)
        
        return selected_featrue_set
    
    def embedded_method(data,target,method):
        '''
        1 基于惩罚项的特征选择法(penalty)        
        2 基于树模型的特征选择法(tree)        
        '''
        
        if method == 'penalty':
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import LogisticRegression
            #带L1惩罚项的逻辑回归作为基模型的特征选择
            selected_featrue_set = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(data, target)
        elif method == 'tree': 
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import GradientBoostingClassifier            
            #GBDT作为基模型的特征选择
            selected_featrue_set = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, target)
            
        return selected_featrue_set

#################### 3
def feature_reduce():
    '''
    减少特征：降维
    降维：主成分分析法（PCA）、线性判别分析法（LDA）
    
    当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大，
    训练时间长的问题，因此降低特征矩阵维度也是必不可少的。
    
    PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能。
    所以说PCA是一种无监督的降维方法，而LDA是一种有监督的降维方法。
    '''
    
    pass

#    # 1 主成分分析法，返回降维后的数据
#    from sklearn.decomposition import PCA
#    #参数n_components为主成分数目
#    PCA(n_components=2).fit_transform(iris.data)
#    
#    # 2 线性判别分析法，返回降维后的数据
#    from sklearn.lda import LDA
#    #参数n_components为降维后的维数
#    LDA(n_components=2).fit_transform(iris.data, iris.target)
    
#################### 4
def feature_scale(data,method):
    '''
    特征无量纲化
    常见的无量纲化方法有标准化和区间缩放法。
    
    标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。
    区间缩放法利用了边界值信息，将特征的取值区间缩放到某个特点的范围，例如[0, 1]等。
    
    标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，
        将样本的特征值转换到同一量纲下。
　　归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，
        拥有统一的标准，也就是说都转化为“单位向量”。
    
    操作流程：先做特征选择，分割训练测试集，再进行这一步
    '''
    
    if method == 'scale':
        from sklearn.preprocessing import scale
        scale(data)
        
    elif method == 'standard': 
        # 1 标准化，返回值为标准化后的数据
        from sklearn.preprocessing import StandardScaler
        StandardScaler().fit_transform(data)
    elif method == 'minmax': 
        # 2 区间缩放，返回值为缩放到[0, 1]区间的数据
        from sklearn.preprocessing import MinMaxScaler
        MinMaxScaler().fit_transform(data)
    elif method == 'normal': 
        #3 归一化，返回值为归一化后的数据
        from sklearn.preprocessing import Normalizer
        Normalizer().fit_transform(data)    
    
#################### main part
def processed_featrue_set(ML):
    '''
    
    '''
    # read data
    clean_featrue_set = pd.read_csv(ML['Feature_Process_File'] + '_clean_fea.csv')
    target = clean_featrue_set['Condition']
    data = clean_featrue_set.ix[:,1:]
    
    processed_featrue_set = FeatureSelection.embedded_method(data,target,'tree')
    
    # 合并标签、数据
    processed_featrue_set = pd.DataFrame(target).join(pd.DataFrame(processed_featrue_set),how='outer')
    
    # save 
    if not os.path.isfile(ML['Feature_Process_File'] + '_sel_fea.csv'):
        processed_featrue_set.to_csv(ML['Feature_Process_File'] + '_sel_fea.csv',index = False) 



if __name__ == '__main__':
    processed_featrue_set(ML)





