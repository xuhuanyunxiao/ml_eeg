import os

import numpy as np
import pandas as pd

from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import style
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import time

style.use('ggplot')

'''
EEG_feature_loading.py

'''
os.chdir("./DATAS")
#os.chdir("D:/CCworkFiles/EEG_ML/DATAS")  # use in cml

# 1. Data Loding
#    read all excel files in "./DATAS".
files = os.listdir()
datas = [i for i in files if i[-4:]=='xlsx' and i[0]!="~"]
strN = []
for i in datas:
    if "FeatureLabel" in i:
        label = pd.read_excel(i, sheetname="FeatureLabel", header=None)
        #label = pd.DataFrame(labels_org[0])
        label.columns = ['label_'+str(i+1) for i in np.arange(int(label.shape[1])) ]
    elif "FeatureData" in i:
        features = pd.read_excel(i, sheetname="FeatureData", header=None)
        for i in np.arange(int(features.shape[1])):
            if i<9 :
                strN.append('00'+str(i+1))
            elif i<99 :
                strN.append('0'+str(i+1))
            elif i<999:
                strN.append(str(i+1))
        features.columns = ['F-'+i for i in strN]
#create index for label and features
label['SNum'] = range(1, label.shape[0] + 1)
label = label.set_index('SNum',drop=False)
features['SNum'] = range(1, features.shape[0] + 1)
features = features.set_index('SNum',drop=False)

# 2. Raw Features Summary
# join label with features
#label_features = label.join( features, how='outer')
label_features = pd.merge(label, features, how='outer')
#print(label_features.head())

pre_name = time.strftime("%Y_%m_%d_%H_%M_%S_",time.localtime()) + datas[0][:-16]
def Summary_Grouped_Features():
    grouped_by_label = label_features.groupby('label_1')
    description_of_grouped_features = grouped_by_label.describe().sort_index(axis=1, ascending=True)
    #print(description_of_grouped_features.head())
    #description_of_grouped_features = grouped_by_label.describe(percentiles=[ 0.05*i for i in range(1,20)]).sort_index(axis=1, ascending=True)
    # save
    file_name = pre_name + 'FeaturesDescription.xlsx'
    try:
        output_dir = '../OutPuts/'
        description_of_grouped_features.to_excel(output_dir + file_name,'FeaturesDescription')
    except:
        os.mkdir('../OutPuts/')
        output_dir = '../OutPuts/'
        description_of_grouped_features.to_excel(output_dir + file_name,'FeaturesDescription')
#Summary_Grouped_Features()

# 3. Missing Value and Extreme Value
# 3.1 Missing Value
#   a) drop samples with any NaN
label_features_dropna = label_features.dropna(axis=0, how='any')
#   b) fill NaN with extreme values
#label_features = label_features.fillna(value=-9999)

# 3.2 remove extreme datas


datas = label_features_dropna.iloc[:,1:]
target = label_features_dropna.iloc[:,0]

# 4. Visualization
def Plot_Feature_Hist(df=label_features):
    condition_num = df['label_1'].unique().size
    cname = ['condition'+str(i+1) for i in range(condition_num)]
    for i in range(condition_num):
        cname[i] = df[df['label_1'] == i+1 ]
    pp = PdfPages('../OutPuts/test.pdf')
    for i in df.columns:
        num_bins = 50

        plt.subplot(3,1,1)
        plt.hist(cname[0][i].dropna(), num_bins, histtype='bar',facecolor='green', rwidth=0.8)
        plt.ylabel('condition1')
        plt.title('histgram of '+i)

        plt.subplot(3,1,2)
        plt.hist(cname[1][i].dropna(), num_bins, histtype='bar',facecolor='red', rwidth=0.8)
        plt.ylabel('condition2')

        plt.subplot(3,1,3)
        plt.hist(cname[2][i].dropna(), num_bins, histtype='bar',facecolor='blue', rwidth=0.8)
        plt.ylabel('condition3')
        plt.xlabel('value')

        #plt.show()
        pp.savefig()
    pp.close()

def Plot_Feature_Box(df=label_features):
    condition_num = df['label_1'].unique().size
    try:
        os.chdir('../OutPuts/')
    except:
        os.mkdir('../OutPuts/')
        os.chdir('../OutPuts/')
    pp = PdfPages('feature_plot_box.pdf')

    for i in range(condition_num):
        df[df['label_1'] == i+1 ].drop(['label_1','label_2','label_3','SNum'], axis=1).boxplot(rot = 60)
        plt.title('Condition_'+str(i+1))
        #plt.show()
        pp.savefig(dpi=300)
    pp.close()

def Plot_Feature_3D(df=label_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xs = df['SNum']
    Ys = df.columns
    Zs = df.values

    ax.plot(Xs,Ys,Zs, label='----')
    ax.set_xlabel('samples')
    ax.set_ylabel('features')
    ax.set_zlabel('values')
    ax.legent()

    plt.show()


#Plot_Feature_Hist()
#Plot_Feature_Box()
Plot_Feature_3D()
