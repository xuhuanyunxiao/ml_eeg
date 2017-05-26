# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 19:15:16 2016

@author: yishikeji-01
"""
#%%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# from matplotlib.patches import Polygon
# import matplotlib.ticker as mtick

# import math

#from win32api import GetSystemMetrics
#width = GetSystemMetrics(0)
#height = GetSystemMetrics(1)

#%%
params={
    'axes.labelsize': '25',       
    'xtick.labelsize':'10',
    'ytick.labelsize':'15',
    'lines.linewidth':2 ,
    'legend.fontsize': '15',
    'text.fontsize': '20',
    'figure.figsize'   :'160,90',   # set figure size
    }

pylab.rcParams.update(params)            #set figure parameter

#line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #set line style



#%%
mean = np.floor(20 + 500 * np.random.random((3,20)))  # mean.ravel()
se = np.floor(5 + 10 * np.random.random((3,20)))

M = pd.DataFrame(mean,columns=np.arange(1,21),
                  index = ['Condition_1','Condition_2','Condition_3'])
SE = pd.DataFrame(se,columns=np.arange(1,21),
                  index = ['Condition_1','Condition_2','Condition_3'])

ind = np.arange(1,21)  # the x locations for the groups

# raw data
N = 10000
Cond1_data = np.random.normal(1, 1, N).reshape(500,20)


#%%
fig, ax = plt.subplots()

#%%
# 1 bar
width = 0.20       # the width of the bars
if 0:  
  fig.canvas.set_window_title('A Barplot Example')  
  rects1 = ax.bar(ind - width * 1.5, M.ix['Condition_1'], width, color='b', \
                  yerr=SE.ix['Condition_1'],alpha = 0.3, label = 'Cond_1',)
  rects2 = ax.bar(ind - width * 0.5, M.ix['Condition_2'], width, color='g', \
                  yerr=SE.ix['Condition_2'],alpha = 0.3, label = 'Cond_2')
  rects3 = ax.bar(ind + width * 0.5, M.ix['Condition_3'], width, color='r', \
                  yerr=SE.ix['Condition_3'],alpha = 0.3, label = 'Cond_3')

# 2 line
if 0:
  fig.canvas.set_window_title('A Lineplot Example')  
  ax.plot(ind + width * 1, M.ix['Condition_1'], label = 'Cond_1', color='b',alpha = 0.3)
  ax.plot(ind + width * 2, M.ix['Condition_2'], label = 'Cond_2', color='g',alpha = 0.3)
  ax.plot(ind + width * 3, M.ix['Condition_3'], label = 'Cond_3', color='r',alpha = 0.3)

if 0:  
  fig.canvas.set_window_title('A lineplot Example')
  x = ind
  y11 = M.ix['Condition_1'] + 3 * SE.ix['Condition_1']
  y12 = M.ix['Condition_1'] - 3 * SE.ix['Condition_1']
  y21 = M.ix['Condition_2'] + 3 * SE.ix['Condition_2']
  y22 = M.ix['Condition_2'] - 3 * SE.ix['Condition_2']
  y31 = M.ix['Condition_3'] + 3 * SE.ix['Condition_3']
  y32 = M.ix['Condition_3'] - 3 * SE.ix['Condition_3']

  ax.plot(x, M.ix['Condition_1'], label = 'Cond_1', color='r')
  ax.plot(x, M.ix['Condition_2'], label = 'Cond_2', color='g')
  ax.plot(x, M.ix['Condition_3'], label = 'Cond_3', color='b')
  
  ax.plot(x, y11,'--', x, y12,'--', color='red',linewidth = 0.1)
  ax.fill_between(x, y11, y12, facecolor='red', interpolate=True,alpha = 0.05)
  ax.plot(x, y21,'--', x, y22,'--', color='green',linewidth = 0.1)
  ax.fill_between(x, y21, y22, facecolor='green', interpolate=True,alpha = 0.05)
  ax.plot(x, y31,'--', x, y32,'--', color='blue',linewidth = 0.1)
  ax.fill_between(x, y31, y32, facecolor='blue', interpolate=True,alpha = 0.05)


# 3 box
if 1:
    fig.canvas.set_window_title('A Boxplot Example')
    handle = ax.boxplot(Cond1_data, notch='True', sym='+', vert=1, whis=1.5)
    
    plt.setp(handle['boxes'], color='black')
    plt.setp(handle['whiskers'], color='black')
    plt.setp(handle['fliers'], marker='+', color='red')

#%%
# plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
  
#%%
ax.set_title('Scores by Condition') 

# label
ax.set_ylabel('Power')
ax.set_xlabel('Feature')

#%% Set the axes ranges and axes labels
ax.set_xlim(0,21)

ax.set_xticks(ind)

# tickname
tick = range(1,len(M.ix['Condition_1'])+1)
ticklabel = []
for i in tick:ticklabel.append('Fea_' + str(i)) 
ax.set_xticklabels(ticklabel)

#%%
ax.grid(True)

## y grid
#ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#               alpha=0.5)
#ax.set_axisbelow(True) # Hide these grid behind plot objects
#%%
ax.legend(loc="upper right") 

#%%

ax.text(ax.get_xlim()[1] * 0.6, ax.get_ylim()[1] * 0.9,
            'nums of conditon')

#%%
# set the global default figure size to be 10 x 10
plt.rc('figure',figsize=(10,10))

#%%
plt.show()


#%% save
# .pdf  .png  .ps  .eps
# plt.savefig（figpath.svg')









