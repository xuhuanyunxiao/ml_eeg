# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:37:03 2016

@author: yishikeji-01
"""

#%%
import numpy as np
import pandas as pd 
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from matplotlib import pylab

import datetime

from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc

#%% 一、可视化
# 定义股票开始日期和截止日期（系统今日时间），并获取数据
start = datetime.datetime(2010,1,1)
end = datetime.date.today()
maotai = web.DataReader('600519.SS','yahoo',start,end)

# 贵州茅台股票基本数据
print(type(maotai))
print(len(maotai))
print(maotai.head())
print(maotai.tail())
# 每月一次的，開盤(Open)、最高(High)、最低(Low)、收盤(Close)、
# 平均成交量(Avg Vol)以及調整後的收盤價(Adj Close)

# 沪市A股贵州茅台的近7年的股价数据
#pylab.rcParams['figure.figsez'] = (15,9)
# 可视化一：单只
maotai['Adj Close'].plot(grid = True)

#%% 可视化二：单只
# 阴阳烛图candlestick_ohlc
def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    # dayFormatter = DateFormatter('%d')      # e.g., 12
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
    ax.grid(True)
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(pylab.date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

pandas_candlestick_ohlc(maotai)

#%% 多只股票的可视化如何表现呢？
# 增加同仁堂600085.SS和全聚德002186.SZ两只股票
QJD = web.DataReader('002186.SZ','yahoo',start,end) # 全聚德
TRT = web.DataReader('600085.SS','yahoo',start,end) # 同仁堂

# 将三只股票聚合到一个DataFrame中
stocks = pd.DataFrame({'QJD':QJD['Adj Close'],
                       'TRT':TRT['Adj Close'],
                       'maotai':maotai['Adj Close']})

print(stocks.head())

#%% 可视化三：多只
# 走势图
stocks.plot(grid = True)

# 因为三只股票的股价不同，坐标尺度不统一造成可视化不清晰。
# 考量不同股票的股价不同，坐标标度有显著尺度差异，可用考虑用双轴标度可视化。
stocks.plot(secondary_y = ['TRT','QJD'],grid = True)

# 坐标尺度考虑是一方面，我们也可以考虑采用基期2010-1-4日股价作为基点
# 进行对标或标准化股票指数化。
# 趋势图
stock_return = stocks.apply(lambda x:x / x[0])
print(stock_return.head())
stock_return.plot(grid = True).axhline(y = 1, color = 'black', lw = 2)

# 还可以计算各种股价的增长或变化
# 先展示用log变化：
stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
print(stock_change.head())
stock_change.plot(grid = True).axhline(y = 0, color = 'black', lw = 2)

# 移动平均的趋势图
# 在Python中实现q-day的移动平均（moving average）是比较方便的
# 分别指定20d、50d和200d的窗口期分析移动平均的走势图
maotai['20d'] = np.round(maotai['Close'].rolling(window = 20, center = False).mean(), 2)
pandas_candlestick_ohlc(maotai.loc['2016-01-04':'2016-12-13',:], otherseries = '20d')

maotai['50d'] = np.round(maotai['Close'].rolling(window = 50, center = False).mean(), 2)
maotai['100d'] = np.round(maotai['Close'].rolling(window = 100, center = False).mean(), 2)
pandas_candlestick_ohlc(maotai.loc['2016-01-04':'2016-12-13',:], otherseries = ['20d','50d','100d'])

####################################################################


####################################################################
#%% 二、确定Buy-Sell时间点模型
# 以20d和50d天的移动平均作为案例进行预测建模
start = datetime.datetime(2016,1,1)
end = datetime.date.today()
gzmt = web.DataReader('600519.SS','yahoo',start,end)  # 贵州茅台

print(type(gzmt))
print(gzmt.tail())

#%%
# 计算该只股票的20d和50d的移动平均，20d我们称为短线-快线，50d我们称为长线-慢线
gzmt['20d'] = np.round(gzmt['Close'].rolling(window = 20, center = False).mean(), 2)
gzmt['50d'] = np.round(gzmt['Close'].rolling(window = 50, center = False).mean(), 2)

pandas_candlestick_ohlc(maotai.loc['2016-01-01':'2016-12-21',:], otherseries = ['20d','50d'])

# 计算20d与50d移动平均值的差值
gzmt['20d-50d'] = gzmt['20d'] - gzmt['50d']
print(gzmt.tail())

# 通过判别20d-50d的差值给出判断状态：
# （Regime：1=牛市看涨，0=不交易，-1=熊市看跌）
gzmt['Regime'] = np.where(gzmt['20d-50d'] > 0, 1, 0)
gzmt['Regime'] = np.where(gzmt['20d-50d'] < 0, -1, gzmt['Regime'])
gzmt.loc['2016-01-01':'2016-12-21','Regime'].plot(ylim = (-2,2)).axhline(y = 0, color = 'black', lw = 2)

# 计算一下看到，贵州茅台股票在2016年出现了166日牛市，36个日熊市，49日平盘的趋势预测。
print(gzmt['Regime'].value_counts())

#%%
# 定义Buy-Sell的信号，生成Signal列
print(gzmt.tail())

regime_orig = gzmt.ix[-1, 'Regime']
gzmt.ix[-1, 'Regime'] = 0
gzmt['Signal'] = np.sign(gzmt['Regime'] - gzmt['Regime'].shift(1))

gzmt.ix[-1, 'Regime'] = regime_orig
print(gzmt.tail())
# 这里注意两点：1-我们要设定Regime字段的最后一行为零，以保证交易的截止。
# 2-我们让Regime列下行错位一行，用Shift(1)功能，然后Regime与Regime.shift(1)相减表示Signal：
# （Signal：1=Buy，0=不交易，-1=Sell）

# 在251个交易日中，明显出现四个买卖信号点
gzmt['Signal'].plot(ylim = (-2,2))

# 四个信号点的交易情况：
# 基于收盘价，将买卖信号标示成为Buy-Sell
gzmt_signals = pd.concat([
                          pd.DataFrame({'Price':gzmt.loc[gzmt['Signal'] == 1, 'Close'],
                                        'Regime':gzmt.loc[gzmt['Signal'] == 1, 'Regime'],
                                        'Signal':'Buy'}),
                          pd.DataFrame({'Price':gzmt.loc[gzmt['Signal'] == -1, 'Close'],
                                        'Regime':gzmt.loc[gzmt['Signal'] == -1, 'Regime'],
                                        'Signal':'Sell'})])
gzmt_signals.sort_index(inplace = True)
gzmt_signals.ix[-1, 'Signal'] = 'Buy'

print(gzmt_signals)

#%% 如果按照上述策略，在长线投资情况下，选择某日买入至某日卖出后的股价每股盈亏或盈利是：
gzmt_long_profits = pd.DataFrame({
    'Price':gzmt_signals.loc[(gzmt_signals['Signal'] == 'Buy') & 
                             gzmt_signals['Regime'] == 1, 'Price'],
    'Profit':pd.Series(gzmt_signals['Price'] - gzmt_signals['Price'].shift(1).loc[
                      gzmt_signals.loc[(gzmt_signals['Signal'].shift(1) == 'Buy') & 
                    (gzmt_signals['Regime'].shift(1) == 1)].index].tolist()),
    'End Date':gzmt_signals['Price'].loc[
                      gzmt_signals.loc[(gzmt_signals['Signal'].shift(1) == 'Buy') & 
                (gzmt_signals['Regime'].shift(1) == 1)].index].index
    })

print(gzmt_long_profits)

pandas_candlestick_ohlc(gzmt,stick = 45, otherseries = ['20d','50d'])


####################################################################


####################################################################
#%% 三、投资组合
# 600050中国联通，600519贵州茅台，002186全聚德，600085同仁堂，
# 000002万科A，601398工商银行，000917电广传媒，600030中信证券，
# 000027深圳能源，000665湖北广电

# 修正股票价格
def ohlc_adj(dat):
    return pd.DataFrame({'Open':dat['Open'] * dat['Adj Close'] / dat['Close'],
                         'High':dat['High'] * dat['Adj Close'] / dat['Close'],
                         'Low':dat['Low'] * dat['Adj Close'] / dat['Close'],
                         'Close':dat['Adj Close']})

gzmt_adj = ohlc_adj(gzmt)

#%%








































