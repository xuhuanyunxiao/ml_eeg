# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:23:32 2017

@author: yishikeji-01
"""

# 《Python 金融大数据分析》 第六章

####################################################################

# 6.2 金融数据

####################################################################
#%%
import math
import numpy as np
import pandas as pd
import pandas_datareader.data as web
#%% 从雅虎财经读取德国DAX指数股票信息
DAX = web.DataReader(name = '^GDAXI', data_source = 'yahoo', start = '2000-1-1')

DAX.info()
DAX.head() # 前五行
DAX.tail() # 后五行

DAX['Adj Close'].plot()
#%% 计算对数收益率
DAX['Ret_Loop'] = 0.0
for i in range(1,len(DAX)):
    DAX['Ret_Loop'][i] = np.log(DAX['Close'][i] / DAX['Close'][i - 1])

# 同上面的for循环，并且更快更简洁
DAX['Return'] = np.log(DAX['Close'] / DAX['Close'].shift(1))
    
DAX[['Close','Ret_Loop']].tail()
DAX['Adj Close'].plot()
del DAX['Ret_Loop']

#%%
# DAX指数和每日指数收益
DAX[['Close','Return']].plot(subplots = True, style = 'b', figsize = (8,5))

# 移动平均值
DAX['42d'] = pd.rolling_mean(DAX['Close'], window = 42)
DAX['252d'] = pd.rolling_mean(DAX['Close'], window = 252)

DAX[['Close','42d','252d']].tail()
# DAX指数及移动平均值
DAX[['Close','42d','252d']].plot()

#%% 对数收益率的移动历史标准差，及移动历史波动率
# moving annual volatility
DAX['Mov_Vol'] = pd.rolling_std(DAX['Return'], window = 252) * math.sqrt(252)
# DAX指数和移动年化波动率
DAX[['Close','Mov_Vol','Return']].plot(subplots = True, style = 'b', figsize = (8,5))

#%%
####################################################################

# 6.3 回归分析

####################################################################
#%%
from urllib import urlretrieve

#%%






































