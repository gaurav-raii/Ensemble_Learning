# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:05:26 2019
@author: gaura
"""
import numpy as np
import pandas as pd
#importing the excel file
df=pd.read_excel("oilproduction.xlsx")
# encoding the categorical data
def my_encoder(z):
    for i in z:
        a=df[i][df[i].notnull()].unique()
        for col_name in a:
            df[i+'_'+str(col_name)]= df[i].apply(lambda x: 1 if x==col_name else 0)
categorical = ['Operator','County']
my_encoder(categorical)
df= df.drop(columns=categorical)

X= np.asarray(df.drop(columns="Log_Cum_Production"))
Y= df["Log_Cum_Production"]
Y = np.asarray(Y)

# Fitting Decision trees  
from sklearn.model_selection import cross_val_score
score_list=['neg_mean_squared_error','neg_mean_absolute_error']
search_depths=[3,4,5,6,7,8,9,10,11,12,13,14,15]

Decision_Tree_Metrics_Table= pd.DataFrame(index=range(13),columns= score_list)
from sklearn.tree import DecisionTreeRegressor
k=0
for d in search_depths:
    dtc= DecisionTreeRegressor(criterion="mse", max_depth=d, min_samples_split=5,min_samples_leaf=5)
    mean_score=[]
    std_score=[]
    for s in score_list:
        dtc_4 = cross_val_score(dtc,X,Y,scoring=s, cv=4)
        mean= dtc_4.mean()
        Decision_Tree_Metrics_Table.loc[k,s] = mean
    k=k+1
    
Decision_Tree_Metrics_Table = Decision_Tree_Metrics_Table.assign(Depths= search_depths)
for k in range(13):
    for s in score_list:
        Decision_Tree_Metrics_Table.loc[k,s]= -(Decision_Tree_Metrics_Table.loc[k,s])
Decision_Tree_Metrics_Table .rename(columns={'neg_mean_squared_error': 'mean_squared_error', 'neg_mean_absolute_error': 'mean_absolute_error'}, inplace=True)        

#best depth = 7 for decision trees  as it has the least cross validation mean abolute error 
 dtr= DecisionTreeRegressor(criterion="mse", max_depth=7, min_samples_split=5,min_samples_leaf=5)
 dtf = dtr.fit(X, Y) 
from AdvancedAnalytics import DecisionTree
DecisionTree.display_metrics(dtf, X, Y)


#using RandomForest 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, n_jobs=1, random_state=12345)
rfr = rfr.fit(X, Y)  
from AdvancedAnalytics import DecisionTree
DecisionTree.display_metrics(rfr, X, Y)