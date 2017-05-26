# -*- coding: utf-8 -*-

#%%

import numpy as np
import pandas as pd

from cycler import cycler
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from matplotlib.pyplot import savefig
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import MultiCursor
import matplotlib.dates as mdate
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

#ChineseFont1 = FontProperties(fname = 'C:\\Windows\\Fonts\\simsun.ttc')
ChineseFont2 = FontProperties('SimHei')

#%%
def plot_3d(datas,label,file_path,ConditionName,thresh,
            viewangle=[15, 45] ,
            title = '1 Import EEG'):
    
    for i in range(len(ConditionName)):
        data = []
        data = datas.ix[label == i+1]

        fig = plt.figure(figsize=(40,30))
        ax = fig.gca(projection='3d')
        ax.view_init(viewangle[0],viewangle[1]) # 视角(上下，左右)

        X = np.arange(1,data.shape[1] + 1)
        Y = np.arange(1,data.shape[0] + 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.array(data)
        
        x = np.linspace(0,data.shape[1]+100,50)
        y = np.linspace(0,data.shape[0]+500,100)
        x, y = np.meshgrid(x, y)
        Z1 = np.ones_like(x) * thresh
        Z2 = np.ones_like(x) * 0
        Z3 = np.ones_like(x) * (-thresh)

        # Plot the surface.       
        ax.plot_surface(x, y, Z1,alpha=0.1,linewidth=2,)
        ax.plot_surface(x, y, Z2,alpha=0.1,linewidth=2,)
        ax.plot_surface(x, y, Z3,alpha=0.1,linewidth=2,)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True) 
        
        ax.set_xlim(0, data.shape[1]+100)
        ax.set_ylim(0, data.shape[0]+500)
        ax.set_zlim(-250, 250)
        
        coord = [ax.get_xlim3d()[1],ax.get_ylim3d()[1],ax.get_zlim3d()[1]]
        ax.set_xlabel(u'时间轴(s)', fontproperties = ChineseFont2,fontsize = 30)
        ax.set_ylabel(u'样本数轴(个)',fontproperties = ChineseFont2,fontsize = 30)
        ax.set_zlabel(u'幅值（微伏）',fontproperties = ChineseFont2,fontsize = 30)
        
        ax.set_xticks(np.arange(0,data.shape[1],50))
        ax.set_yticks(np.arange(0,data.shape[0],1000))
        ax.set_zticks(np.arange(-250,250,50))
        
        # title
        ax.set_title(title,fontsize = 50)
        
        # text
        count_s = 0
        for z in Z:
            if ((max(z) > thresh) | (min(z) < -thresh)):
                count_s +=1
        labels = ['样本数：%d' % (data.shape[0]),'±%s外：%0.02f' % (str(thresh),count_s/Z.shape[0]),
                  '最大值：%d' % (Z.max()),'最小值：%d' % (Z.min()),
                  '均值：%d' % (Z.mean()),'标准差：%d' % (Z.std())]        
        for labe,la in zip(labels,range(len(labels))):
            ax.text(coord[0] *0.95,coord[1] *0.01, coord[2] *(0.9-la*0.2), labe, 
                    color='white',fontproperties = ChineseFont2,fontsize = 30,
                    bbox={'facecolor':'black', 'alpha':0.2, 'pad':10})
        ax.text2D(0.05, 0.85, 'Condition: %s' %ConditionName[i], 
                  fontsize = 40,transform=ax.transAxes)
        
        # 显示colorbar
        cbar = fig.colorbar(surf, ax=ax,shrink=0.5, aspect=10, spacing = 'uniform') # 
        cbar.set_label('微伏',fontproperties = ChineseFont2,fontsize = 30)
        cbar.set_ticks(np.arange(-150,200,50))
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticklabels(tuple([str(ticklabels) for ticklabels in np.arange(-150,200,50)]))

        plt.show()
            
        # save
        filename_fig = file_path + '_' + title + '_' + ConditionName[i] + '.jpg'
        savefig(filename_fig)

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
        
        'lines.linewidth' :4,
        'lines.markersize':15,
        
        'legend.fontsize':30,
        
        'image.cmap':'Blues',
        'image.interpolation':'nearest',
   
        'figure.figsize' : '40, 30'    # set figure size
    }
pylab.rcParams.update(params)

#%%
def plot_day_count_ratio(data1,data2,data3,file_path,title = "每日数据"): # 比率、计数
    fig = plt.figure()
    alpha = 0.2
    
    def plot_main(ax,data,day_line,legendlabel,flag = 'ratio'):
        j = 0        
        for i in data.ix[:,1:]:
            j +=1
            if flag == 'ratio':
                ax.plot(data[i],label = legendlabel[j]+'(%.2f)' % data.mean(axis=0)[j])
            elif flag == 'count':
                ax.plot(data[i],label = legendlabel[j]+'(%d)' % data.sum(axis=0)[j])
                
        if flag == 'ratio':
            ax.axhline(y=day_line, color='black',label = 'day average loss ratio (%.2f)' % day_line)            
            ax.plot(data.ix[:,0],'mo--',label=legendlabel[0]+'(%.2f)' % data.mean(axis=0)[0])
            ax.set_ylim([0,1])            
            ax.set_ylabel('损失比率',fontproperties = ChineseFont2,fontsize = 30)
            #ax2.yaxis.set_label_coords(-0.1, 0.5) # y label 位置 
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # 主刻度间隔 
            ax.set_xticklabels([]) # 主刻度label
            ax.set_yticks(np.arange(0,1.1,0.2)) 
        elif flag == 'count':
            ax.set_xlabel('日期（天）',fontproperties = ChineseFont2,fontsize = 30)  
            ax.set_ylabel('各条件数据量（个）',fontproperties = ChineseFont2,fontsize = 30)
            # ticks : 时间格式 # ax2.get_xticks()     
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # 主刻度间隔               
            ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m-%d'))#设置时间标签显示格式
            ax.xaxis_date()
            for label in ax.xaxis.get_ticklabels():label.set_rotation(30) # 旋转
            ax.tick_params(axis ='both', which='major', length=8, labelsize =20)
            # 添加直线
            ax.axhline(y=day_line, color='black',label = 'day average count(%d)' % day_line)
            # text
            if data.mode().empty:
                mode = 0
            else:
                mode = data.mode()[0]
            textlabel =  'Total samples:M(%d) SD(%d) mode(%d) meaian(%d) max(%d) min(%d)'\
                             % (data.mean()[0],data.std()[0],mode,
                                data.median()[0],data.max()[0],data.min()[0])
            coord = [ax.get_xlim()[0],ax.get_ylim()[0]]
            ax.text(coord[0],coord[1],textlabel,
                    color='white',fontsize = 30,
                    bbox={'facecolor':'black', 'alpha':alpha, 'pad':10})
            
        # 图例 # 1 右上 2 左上 3 左下 4 右下
        legend = ax.legend(shadow=True,loc=2)
        leg  = legend.get_frame()
        leg.set_facecolor('#e4d8c0') # background
        leg.set_alpha(alpha)
        
        plt.grid(True,which = 'major')
        # 添加区域
        specific_index = np.argwhere(np.array(data.values) == 0) 
        x_index_time = list(set([data.index[specific_index[i][0]] for i in range(len(specific_index))])) # list 去重 set
        if x_index_time:
            x_index_num = mdates.date2num(pd.Index(x_index_time).to_pydatetime())
            #dd = mdates.num2date(x_index_num)
            for ix in x_index_num:ax.axvspan(ix-0.5, ix+0.5, facecolor='#b7ebd9', alpha=alpha)
        else:
            x_index_num = ax.get_xticks()[2]
            ax.axvspan(x_index_num-0.5, x_index_num+0.5, facecolor='#b7ebd9', alpha=alpha)        
            
    ax1 = plt.subplot(211) # 比率 
    day_line = (data3.sum()[0]/data2.sum()[0])/data2.shape[0]
    legendlabel = list(data1.columns)
    plot_main(ax1,data1,day_line,legendlabel,flag = 'ratio')
    
    ax2 = plt.subplot(212) # 计数
    day_line = data2.sum()[0]/(data2.shape[0]*(data2.shape[1]-1))        
    legendlabel = list(data2.columns)
    plot_main(ax2,data2,day_line,legendlabel,flag = 'count')
    
    
    ax3 = ax2.twinx()
    ax3.plot(data2.ix[:,0],'mo--',label=legendlabel[0]+'(%d)' % data2.sum(axis=0)[0])
    ax3.set_ylabel('每天总数据量（个）',fontproperties = ChineseFont2,fontsize = 30)
    legend = ax3.legend(shadow=True,loc=1)
    legend.get_frame().set_facecolor('#e4d8c0') # background    
    legend.get_frame().set_alpha(alpha)
    
    ax3.grid(True,which = 'major',linestyle='-',color = 'blue')  # 网格          

    # 调整图形位置：左下（0,0）  右上（1,1）
    plt.subplots_adjust(bottom=0.1)
    
    ax1.set_title(title+'(共%s天)' % str(data2.index.shape[0]),
                  fontproperties = ChineseFont2,fontsize = 50)# 标题
    multi  = MultiCursor(fig.canvas,(ax1,ax2),color='r',lw=1) # 俩子图一条直线

    plt.show()    
        
    # save
    filename_fig = file_path + '_' + title+ '.jpg'
    savefig(filename_fig)
    
#%%
def get_count_ratio_data(data,datalabel,ConditionName,DayName,thresh):
    # prepare
    label = datalabel.join(pd.DataFrame([DayName[int(day) - 1][3:] for day in datalabel.ix[:,1]],columns =['date']))
    label.rename(columns={'0':'Cond', '1':'Day', '2':'day_file_N'}, inplace=True) 
    # label.index = label['date']
    count_s = []
    for z in np.array(data):
        if ((max(z) > thresh) | (min(z) < -thresh)):
            count_s.append(0)
        else:
            count_s.append(1)
    
    label['quality'] = count_s
    # label = label.sort_index()
    
    # statistics
    def gen_column_name(ConditionName):
        count_name = []
        ratio_name = []
        for i in ConditionName:
            count_name.append(i + '_count')
            ratio_name.append(i + '_loss_ratio')    
        return count_name,ratio_name
        
    count_name,ratio_name = gen_column_name(ConditionName)
    label_stats_count = pd.DataFrame(label['Day'].groupby([label['date'], 
                                                          label['Cond']]).count()).unstack()
    label_stats_count.columns = count_name
    count_sum = label_stats_count.sum(axis = 1)
    label_stats_count.insert(0,'count',count_sum)
    label_stats_count.index = pd.to_datetime(label_stats_count.index)
    
    label_stats_loss = pd.DataFrame(label['Day'].groupby([label['date'], 
            label['Cond'],label['quality'][label['quality']==0]]).count()).unstack().unstack()
    label_stats_loss.columns = ratio_name
    loss_sum = label_stats_loss.sum(axis = 1)
    label_stats_loss.insert(0,'loss_ratio',loss_sum)
    label_stats_loss_ratio = pd.DataFrame(np.array(label_stats_loss)/np.array(label_stats_count),
                                          index = label_stats_loss.index,
                                          columns = label_stats_loss.columns)
    label_stats_loss_ratio.index = pd.to_datetime(label_stats_loss_ratio.index) # index 变为时间格式
    
    return label_stats_loss_ratio,label_stats_count,label_stats_loss
    
#%%
def view_imported_eeg(ML,thresh = 200):
    
    # prepare data and label
    eeg_data_label = pd.read_csv(ML['EEG_Import_File'] + '.csv') 

    eeg_data = eeg_data_label.ix[:,3:]
    eeg_label = eeg_data_label.ix[:,0:3] # Cond, Day, day_file_N
    
    ConditionName = ML['ConditionName']
    file_path = ML['Statis_EEG_Import_File']

    plot_3d(eeg_data,eeg_label.ix[:,0],file_path,ConditionName,thresh)
    
    DayName = ML['DayName']
    label_stats_loss_ratio,label_stats_count,label_stats_loss = get_count_ratio_data(eeg_data,
                                                                    eeg_label,ConditionName,DayName,thresh)
    plot_day_count_ratio(label_stats_loss_ratio,label_stats_count,
                         label_stats_loss,file_path,
                         title = "每日数据")
    # save
    filename_result = ML['Statis_EEG_Import_File'] + '.xlsx'
    with pd.ExcelWriter(filename_result) as writer:
        label_stats_loss_ratio.to_excel(writer,sheet_name='import eeg',startrow=0, startcol=0) 
        label_stats_count.to_excel(writer,sheet_name='import eeg',startrow=6, startcol=0)
        label_stats_loss.to_excel(writer,sheet_name='import eeg',startrow=12, startcol=0)

#%%
def view_processed_eeg(ML,thresh = 200):
    import pickle
    
    # read data
    filename_data = ML['EEG_Prepro_File'] + '_clearn_data.txt'
    clearn_data = pickle.load(open(filename_data, 'rb'))
    filename_label = ML['EEG_Prepro_File'] + '_clearn_label.txt'
    eeg_label = pickle.load(open(filename_label, 'rb')) 

    eeg_data = pd.DataFrame(clearn_data.get_data()[:,0])
    
    ConditionName = ML['ConditionName']
    file_path = ML['Statis_EEG_Prepro_File']

    plot_3d(eeg_data,pd.Series(eeg_label),file_path,ConditionName,thresh,
            title = '2 Processed EEG')
    
#    DayName = ML['DayName']
#    label_stats_loss_ratio,label_stats_count,label_stats_loss = get_count_ratio_data(eeg_data,
#                                                                    pd.DataFrame(eeg_label),ConditionName,DayName,thresh)
#    plot_day_count_ratio(label_stats_loss_ratio,label_stats_count,
#                         label_stats_loss,file_path,
#                         title = "每日数据")
    # save
    filename_result = ML['Statis_EEG_Prepro_File'] + '.xlsx'
    with pd.ExcelWriter(filename_result) as writer:
        eeg_data.to_excel(writer,sheet_name='processed eeg',startrow=0, startcol=0)        