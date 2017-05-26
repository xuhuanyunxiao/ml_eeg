# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:00:31 2017

@author: yishikeji-01
"""

# - 《数据科学实战手册》（R + Python） Tony  人民邮电
# - 第八章 社交网络分析

#%%
import networkx as nx
import unicodecsv as csv

def graph_from_csv(path):
    graph = nx.Graph(name = 'Heroic Social Network')
    with open(path,'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            graph.add_edge(*row)
    return graph




#%%






