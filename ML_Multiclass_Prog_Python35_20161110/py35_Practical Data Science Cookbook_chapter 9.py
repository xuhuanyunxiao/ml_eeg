# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:30:26 2016

@author: yishikeji-01
"""

#%%
import os 

import csv

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime

#%%
os.chdir('D:\XH\\analysis_prog\ML_Multiclass_Prog_Python35_20161110\Practice \
Data Science Cookbook_Code & Data\Chapter09\data')

#%% ###################### 一、加载电影评分数据
#%%
def load_reviews(path,**kwargs):
    '''
    Loads Movielens reviews
    '''
    options = {'fieldnames':('userid','movieid','rating','timestamp'),
               'delimiter':'\t'}
    options.update(kwargs)
    
    parse_date = lambda r,k: datetime.fromtimestamp(float(r[k]))
    parse_int = lambda r,k: int(r[k])
    
    with open(path,'rb') as reviews:
        reader = csv.DictReader(reviews,**options)
        for row in reader:
            row['movieid'] = parse_int(row,'movieid')
            row['useid'] = parse_int(row,'useid')
            row['rating'] = parse_int(row,'rating')
            row['timestamp'] = parse_date(row,'timestamp')
            yield row
            
def relative_path(path):
    '''
    Return a path relative from this file
    '''
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname,path)
    return os.path.normpath(path)

def load_movies(path,**kwargs):
    '''
    load movielens movies
    '''
    options = {'fieldnames':('movieid','title','release','video','url'),
               'delimiter': '|',
               'restkey':'genre'}
    options.update(kwargs)    
    
    parse_int = lambda r,k: int(r[k])
    parse_date = lambda r,k: datetime.strptime(r[k],'%d-%b-%Y') if r[k] else None
                                            
    with open(path,'rt') as movies:
        reader = csv.DictReader(movies,**options)
        for row in reader:
            row['movieid'] = parse_int(row,'movieid')
            row['release'] = parse_date(row,'release')
            row['video'] = parse_date(row,'video')
            yield row                                          

class MovieLens(object):
    '''
    Data structure to build our recommender model on.
    '''
    
    def __init__(self,udata,uitem):
        '''
        Instantiate with a path to u.data and u.item
        '''
        self.udata = udata
        self.uitem = uitem
        self.movies = {}
        self.reviews = defaultdict(dict)
        self.load_dataset()
        
    def load_dataset(self):
        '''
        load the two datasets into memory, indexed on the ID.
        '''
        for movie in load_movies(self.uitem):
            self.movies[movie['movieid']] = movie
        
        for review in load_reviews(self.udata):
            self.reviews[review['useid']][review['movieid']] = review

data = relative_path('ml-100k/u_data.txt')
item = relative_path('ml-100k/u_item.txt')
model = MovieLens(data,item)      


#%%






#%%






#%%






#%%






#%%






