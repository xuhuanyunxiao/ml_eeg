# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:34:40 2016

@author: yishikeji-01
"""

#%%
import numpy as np
import pandas as pd
import scipy as sp

import statsmodels.api as sm
import matplotlib.pyplot as plt

#%% 1 生成数据
# 数据描述：三组被试，每人的数据包括年龄、学历、测试分数、正确率、测试时长
np.random.seed(0)
SampleSize = 1000
degree = ['本科','本科','硕士','硕士','博士']*int(SampleSize / 5)
np.random.shuffle(degree)
data = {'编号':np.arange(1,SampleSize + 1),
        '组别':np.random.randint(1,4,size=SampleSize),
        '年龄':np.random.randint(18,32, size=SampleSize), # 整数
        '学历':degree,
        '测试分数':np.random.normal(75,10,SampleSize),
        '正确率':np.random.uniform(0.65,0.95,SampleSize),
        '测试时长':np.random.randint(20 * 60,30*60,SampleSize), # s
        }
Data = pd.DataFrame(data,
                    columns = ['组别','年龄','学历','测试分数','正确率','测试时长'],
                    index = data['编号'])

#%% 2 数据中加入噪音
# 2.1 加入重复：'测试分数'
Data['测试分数'][np.random.randint(1,101,13)] = 84

# 2.2 加入缺失：'学历'
Data['学历'][np.random.randint(1,101,14)] = np.NaN

# 2.3 加入异常值：'年龄'
Data['年龄'][np.random.randint(1,101,7)] = np.random.randint(10,18, size=7)
Data['年龄'][np.random.randint(1,101,8)] = np.random.randint(32,60, size=8)

#%% 3 数据清洗
# 3.1 缺失数据处理
# http://www.cnblogs.com/sirkevin/p/5767532.html
# pandas使用isnull()和notnull()函数来判断缺失情况。
# 对于缺失数据一般处理方法为滤掉或者填充。
## 3.1.1 滤除缺失数据
# 对于一个Series，dropna()函数返回一个包含非空数据和索引值的Series
# 对于DataFrame，dropna()函数同样会丢掉所有含有空元素的数据
# 但是可以指定how='all'，这表示只有行里的数据全部为空时才丢弃
Data.dropna(how = 'any',inplace = True) #   axis = 1
## 3.1.2 填充缺失数据
# 如果不想丢掉缺失的数据而是想用默认值填充这些空洞，可以使用fillna()函数
# 如果不想只以某个标量填充，可以传入一个字典，对不同的列填充不同的值
# Data.fillna(9999)

# 3.2 去除重复
# data.duplicated() # 返回一个布尔型Series，表示各行是否是重复行
Data.drop_duplicates(['测试分数'],inplace = True) # 返回移除了重复行的DataFrame

# 3.3 修改异常值，用平均年龄代替
MeanValue = Data['年龄'][(Data['年龄'] < 32) & (Data['年龄'] > 17)].mean()
Data['年龄'].replace(Data['年龄'][(Data['年龄'] > 31) | (Data['年龄'] < 18)],
     int(MeanValue),inplace = True)

# 3.4 随机采样
TakeSize = 0.5 * SampleSize # 两个样本相等，则重排序；TakeSize > SampleSize,增加；否则，挑选
sampler = np.random.permutation(int(TakeSize)) # 随机重排序
Data = Data.take(sampler, is_copy=False)

# 3.5 排序
Data.sort_values(by=['组别','年龄','测试时长'],inplace = True)

#%% 4 描述统计
# 4.1 集中趋势
GroupbyCount = Data['学历'].groupby([Data['组别'],Data['学历']]).count()
GroupbySum = Data['组别'].groupby(Data['组别']).sum()
print(GroupbyCount)
print(GroupbySum)

Groupby = Data[['年龄','测试分数','正确率','测试时长']].groupby(Data['组别'])
print(Groupby.mean())
print(Groupby.median())

print(Groupby.describe())

## 
Statc = pd.DataFrame()
Statc_name = ['_mean','_std','_max','_min','_median','_skew']
for i in range(1,4):
    print(Data[Data['组别'] == i].mean())
    Statc['Group' + str(i) + Statc_name[0]] = Data[Data['组别'] == i].mean()
    Statc['Group' + str(i) + Statc_name[1]] = Data[Data['组别'] == i].std()
    Statc['Group' + str(i) + Statc_name[2]] = Data[Data['组别'] == i].max()
    Statc['Group' + str(i) + Statc_name[2]] = Data[Data['组别'] == i].min()
    Statc['Group' + str(i) + Statc_name[3]] = Data[Data['组别'] == i].median()
    Statc['Group' + str(i) + Statc_name[3]] = Data[Data['组别'] == i].skew()

# 4.2 离散趋势
print(Groupby.std())
print(Groupby.max() - Groupby.min())

# 4.3 其他
print(Groupby.skew())

#%% 显示

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].bar(GroupbySum) # 从 0 到 1 均匀分布
axes[0].set_title("the sum of three group")
axes[1].hist(GroupbyCount) # 标准正态分布
axes[1].set_title("分组总数")

fig.tight_layout()

#%%

from texttable import Texttable

table = Texttable()
table.set_cols_align(["l", "r", "c"])
table.set_cols_valign(["t", "m", "b"])
table.add_rows([ [ "Name Of Person", "Age", "Nickname"],
             ["Mr\nXavier\nHuon", 32, "Xav'"],
             ["Mr\nBaptiste\nClement", 1, "Baby"] ])
print(table.draw() + "\n")

table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['t',  # text
                      'f',  # float (decimal)
                      'e',  # float (exponent)
                      'i',  # integer
                      'a']) # automatic
table.set_cols_align(["l", "r", "r", "r", "l"])
table.add_rows([["text",    "float", "exp", "int", "auto"],
                ["abcd",    "67",    654,   89,    128.001],
                ["efghijk", 67.5434, .654,  89.6,  12800000000000000000000.00023],
                ["lmn",     5e-78,   5e-78, 89.4,  .000000000000128],
                ["opqrstu", .023,    5e+78, 92.,   12800000000000000000000]])
print(table.draw())
 
#%% 5 推断统计----参数检验
# 分布检验


# 5.1 单样本T检验
t, p = sp.stats.ttest_1samp(X_samples, mu) 

# 5.2 独立样本T检验
t, p = sp.stats.ttest_ind(X1_sample, X2_sample)

# 5.3 配对样本T检验


# 5.4 单因素方差分析


# 5.5 多因素方差分析



# 多重比较


# 事后检验


# 5.6 重复测量方差分析


# 5.7 相关分析


# 5.8 回归分析


#%% 6 推断统计----非参数检验
np.random.seed(0) # 先定义个种子
X = sp.stats.chi2(df=5) # 卡方分布
X_samples = X.rvs(100) # 用卡方分布生成 100 个随机数

# 假设我们不知道这些数是通过卡方分布产生的，现在要推断分布的形状，
# 只能采取一些经验的方法，其中一种方法是核分布。

kde = sp.stats.kde.gaussian_kde(X_samples) 
# 高斯核函数，kde 是核函数密度估计，核分布是种非参数估计方法
kde_low_bw = sp.stats.kde.gaussian_kde(X_samples, bw_method=0.25) 
# bw_method 为窗宽参数，该值越小，就越受到数据本身的影响
x = np.linspace(0, 20, 100)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].hist(X_samples, normed=True, alpha=0.5, bins=25)
axes[1].plot(x, kde(x), label="KDE")
axes[1].plot(x, kde_low_bw(x), label="KDE (low bw)")
axes[1].plot(x, X.pdf(x), label="True PDF")
axes[1].legend()
# sns.distplot(X_samples, bins=25, ax=axes[2])

fig.tight_layout()

kde.resample(10) # 使用非参数估计的经验分布来进一步抽样

def _kde_cdf(x):
    return kde.integrate_box_1d(-np.inf, x) # 得到经验分布的累积分布图形

kde_cdf = np.vectorize(_kde_cdf) # 向量化函数

fig, ax = plt.subplots(1, 1, figsize=(8, 3))

# sns.distplot(X_samples, bins=25, ax=ax)
x = np.linspace(0, 20, 100)
ax.plot(x, kde_cdf(x))

fig.tight_layout()

































