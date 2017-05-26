# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:49:01 2016

@author: yishikeji-01
"""


#%%
import numpy as np
import jinja2 
import matplotlib.pyplot as plt

import csv
import os

from numpy import ma as ma
#from datetime import datetime
#%% ###################### 一、导入并熟悉世界各国高收入数据集
#%% 1 load data
os.chdir('D:\XH\\analysis_prog\ML_Multiclass_Prog_Python35_20161110')

file_name = 'income_dist.csv'
with open(file_name,'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

print('num_samples:%s' %len(data))
print('num_varaibles:%s' %len(reader.fieldnames))

#%% 查看所有数据
def dataset(path):
    with open(path,'r') as csvfile:
        reader = csv.DictReader(csvfile) # reader.fieldnames
        for row in reader:
            yield row # 生成器

# 列表推导式、集合、字典推导式            
print(set(row['Country'] for row in dataset(file_name))) # 国家
print(min(set(int(row['Year']) for row in dataset(file_name)))) # 年份
# {row['Country'] for row in dataset(file_name)}

#%% 过滤出美国的数据
filter(lambda row: row['Country'] == 'United States',dataset(file_name))

#%% 可视化：matplotlib
def dataset(path,filter_field = None, filter_value = None):
    with open(path,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        if filter_field:
            for row in filter(lambda row: row[filter_field] == filter_value, reader):
                yield row
        else:
            for row in reader:
                yield row
                
def main(path):
    data = [(row['Year'],float(row['Average income per tax unit'])) \
            for row in dataset(path,'Country','United States')]

    width = 0.35
    ind = np.arange(len(data))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar(ind,list(d[1] for d in data))
    ax.set_xticks(np.arange(0,len(data),4))
    ax.set_xticklabels(list(d[0] for d in data)[0::4],rotation = 45)
    ax.set_ylabel('Income in USD')
    plt.title('U.S. Average Income 1913-2008')
    plt.show()
    
main(file_name)

#%%      
dataset = np.recfromcsv(file_name,skip_header = 1)
print(dataset.size) # 矩阵中元素
print(dataset.shape) # 数组维度

#%%    
names = ['Country','Year']
names.extend(['col%i' %(idx+1) for idx in range(352)])
dtypes = 'S64,i4,' + ','.join(['f8' for idx in range(352)])

dataset = np.genfromtxt(file_name,dtype = dtypes,names=names,delimiter = ',',\
                        skip_header = 1,autostrip = 2)

#%% 清理Na： 掩码数组 masked array
ma.masked_invalid(dataset['col1'])

#%% ###################### 二、分析并可视化美国的高收入数据集
#%%
def dataset(path,country = 'United States'):
    '''
    Extract the data for the country provided. Default is United States.
    '''
    with open(path,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in filter(lambda row: row['Country'] == country,reader):
            yield row
            
def timeseries(data,column):
    '''
    Creates a year based time series for the given column
    '''
    for row in filter(lambda row: row[column],data):
        yield (int(row['Year']),row[column])

def linechart(series,**kwargs):
    fig = plt.figure()
    ax = plt.subplot(111)
    
    for line in series:
        line = list(line)
        xvals = [v[0] for v in line]
        yvals = [v[1] for v in line]
        ax.plot(xvals,yvals)
        
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'labels' in kwargs:
        ax.legend(kwargs.get('labels'))
        
    return fig 
        
def percent_income_share(source):
    '''
    Create Income Share chart
    '''
    columns = (
               'Top 10% income share',
               'Top 5% income share',
               'Top 1% income share',
               'Top 0.5% income share',
               'Top 0.1% income share'
               )

    source = list(dataset(source))
    
    return linechart([timeseries(source,col) for col in columns],
                      labels = columns,
                      title = 'U.S. Percentage Income Share',
                      ylabel = 'Percentage')

percent_income_share(file_name)
plt.show()

#%%
def normalize(data):
    '''
    Normalizes the data set. Expects a timeseries input
    '''
    data = list(data)
    norm = np.array(list(d[1] for d in data), dtype = 'f8')
    mean = norm.mean()
    norm /= mean
    
    return zip((d[0] for d in data),norm)
    
def mean_normalized_percent_income_share(source):
    columns = (
               'Top 10% income share',
               'Top 5% income share',
               'Top 1% income share',
               'Top 0.5% income share',
               'Top 0.1% income share'
               )

    source = list(dataset(source))
    
    return linechart([normalize(timeseries(source,col)) for col in columns],
                      labels = columns,
                      title = 'Mean Normalized U.S. Percentage Income Share',
                      ylabel = 'Percentage')

mean_normalized_percent_income_share(file_name)    
plt.show()

#%%
def yrange(data):
    '''
    Get the range of years from the dataset
    '''
    years = set()
    for row in data:
        if row[0] not in years:
            yield row[0]
            years.add(row[0])


def delta(first,second):
    '''
    Returns an array of deltas for the tow arrays.
    '''
    first = list(first)
    years = yrange(first)
    first = np.array(list(d[1] for d in first), dtype = 'f8')
    second = np.array(list(d[1] for d in second), dtype = 'f8')
    
    # Not for use in writing
    if first.size != second.size:
        first = np.insert(first,[0,0,0,0],[None,None,None,None])
        
    diff = first - second
    return zip(years,diff)

def capital_gains_lift(source):
    '''
    Computes capital gains lift in top incomes percentage over time chart
    '''
    columns = (
               ('Top 10% income share-including capital gains','Top 10% income share'),
               ('Top 5% income share-including capital gains','Top 5% income share'),
               ('Top 1% income share-including capital gains','Top 1% income share'),
               ('Top 0.5% income share-including capital gains','Top 0.5% income share'),
               ('Top 0.1% income share-including capital gains','Top 0.1% income share')
               )

    source = list(dataset(source))
    series = [delta(timeseries(source,a),timeseries(source,b)) for a,b in columns]
    
    return linechart(series,
                      labels = list(col[1] for col in columns),
                      title = 'U.S. Capital Gains Income Lift',
                      ylabel = 'Percentage Difference')

capital_gains_lift(file_name)    
plt.show()    

#%% ###################### 三、进一步分析美国的高收入阶层
#%%
def average_incomes(source):
    '''
    Compares percentage average incomes
    '''
    columns = (
               'Top 10% average income',
               'Top 5% average income',
               'Top 1% average income',
               'Top 0.5% average income',
               'Top 0.1% average income'
               )

    source = list(dataset(source))
    
    return linechart([timeseries(source,col) for col in columns],
                      labels = columns,
                      title = 'U.S. average income',
                      ylabel = '2008 US Dollars')    

average_incomes(file_name)
plt.show()

#%%
def average_top_income_lift(source):
    '''
    Compares top percentage avg income over total avg
    '''
    columns = (
               ('Top 10% average income','Top 0.1% average income'),
               ('Top 5% average income','Top 0.1% average income'),
               ('Top 1% average income','Top 0.1% average income'),
               ('Top 0.5% average income','Top 0.1% average income'),
               ('Top 0.1% average income','Top 0.1% average income')
               )

    source = list(dataset(source))
    series = [delta(timeseries(source,a),timeseries(source,b)) for a,b in columns]
    
    return linechart(series,
                      labels = list(col[0] for col in columns),
                      title = 'U.S. Income Disparity',
                      ylabel = '2008 US Dollars')

average_top_income_lift(file_name)    
plt.show()    

#%%
def stackedarea(series,**kwargs):
    fig = plt.figure()
    axe = fig.add_subplot(111)
    
    fnx = lambda s: np.array(list(v[1] for v in s), dtype = 'f8')
    yax = np.row_stack(fnx(s) for s in series)
    xax = np.arange(1917,2008)
    
    polys = axe.stackplot(xax,yax)
    axe.margins(0,0)
    
    if 'ylabel' in kwargs:
        axe.set_ylabel(kwargs['ylabel'])
    if 'labels' in kwargs:
        legendProxies = []
        for poly in polys:
            legendProxies.append(plt.Rectangle((0,0),1,1,
                                               fc = poly.get_facecolor()[0]))
        axe.legend(legendProxies,kwargs.get('labels'))
    if 'title' in kwargs:
        plt.title(kwargs['title'])
        
    return fig
    
def income_composition(source):
    '''
    Compares income composition
    '''
    columns = ('Top 10% income composition-Wages, salaries and pensions',
               'Top 10% income composition-Dividends',
               'Top 10% income composition-Interest Income',
               'Top 10% income composition-Rents',
               'Top 10% income composition-Entrepreneurial income'
               )
    source = list(dataset(source))
    labels = ('Salary','Dividends','Interest','Rents','Business')
    
    return stackedarea([timeseries(source,col) for col in columns],
                       labels = labels,
                       title = 'U.S. Top 10% Income Composition',
                       ylabel = 'Percentage')

income_composition(file_name)    
plt.show()   

#%% ###################### 四、用jinja2汇报结果
#%%
template = jinja2.Template(u'Greetings,{{name}}!')
template.render(name = 'Mr. Praline')

#%%
# 'template' should be the path to be in the current directory
# as written, it is assumed to be in the current directory
jinjaenv = jinja2.Environment(loader = jinja2.FileSystemLoader('templates'))
template = jinjaenv.get_template('report.html')

template.stream(items = ['a','b','c'], name = 'Eric').dump('report-2013.html')

#%%
