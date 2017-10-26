# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:19:58 2017

@author: Makhtar Ba
"""

import numpy 
import sklearn
import pandas as pd
import os
import datetime
from sklearn.decomposition import *
import matplotlib.pyplot as plt

datetime.datetime.strftime()

os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')

with open('Tickers.txt') as tickers:
    reader=tickers.read().split("\n")
    list_tickers=[read for read in reader]
print(list_tickers)

data=pd.read_csv('AAPL.txt'.format(ticker), sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]


data=data[" OPEN"]
data=data.to_frame()
for ticker in list_tickers[2:]:        
    df = pd.read_csv('{}.txt'.format(ticker), sep=",")
    df['DATE']=df['DATE'].apply(lambda x : str(x))
    df['DATE']=df['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))
    df.index=df["DATE"]
    df=df[" OPEN"]
    data=data.merge(df.to_frame(),left_index=True,right_index=True)
    #data=data.merge(data,df)
    

data=data.transpose()
# replicate Data from question in DataFrame


def scatterplot(x_data, y_data, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.autofmt_xdate()

#use column headers as x values
x = data.columns
# sum all values from DataFrame along vertical axis
y = data.values
scatterplot(x,y, "x_label", "y_label", "title")

plt.show()
    
