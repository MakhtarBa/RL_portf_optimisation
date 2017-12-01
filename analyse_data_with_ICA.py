# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:19:58 2017

@author: Makhtar Ba
"""

import numpy as np
import sklearn
import pandas as pd
import os
import datetime
from sklearn.decomposition import *
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA



#del data_df
#del data

os.chdir('C:/Users/saumy/Google Drive/data')

with open('Tickers.txt') as tickers:
    reader=tickers.read().split("\n")
    list_tickers=[read for read in reader]

data=pd.read_csv('AAPL.txt', sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]


data=data[" OPEN"]
data_df=pd.DataFrame(data)

print(type(data_df))
damaged_stocks=[]
for ticker in list_tickers[2:]:        
    df = pd.read_csv('{}.txt'.format(ticker), sep=",")
    df['DATE']=df['DATE'].apply(lambda x : str(x))
    df['DATE']=df['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))
    df.index=df["DATE"]
    df = df[" OPEN"]
    if len(df)<3000:
        #print(ticker)
        damaged_stocks.append(ticker)        
    else :
        #print(type(df))
        data_df=data_df.merge(pd.DataFrame(df),left_index=True,right_index=True)
    #data=data.merge(data,df)
list_tickers = [x for x in list_tickers if x  not in damaged_stocks]
data_df.columns=list_tickers[1:]
data_df=data_df.transpose()
# replicate Data from question in DataFrame


def scatterplot(x_data, y_data, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.autofmt_xdate()

#use column headers as x values
x = data_df.columns
# sum all values from DataFrame along vertical axis
y = data_df.loc['AAPL']
dates=data_df.columns

lag=[(dates[i]-dates[i+1]).days for i in range (len(dates)-1)]
lag_dataframe = pd.DataFrame(lag)
lag_dataframe.plot(kind='hist')

plt.show()


##########################################
ica = FastICA(n_components=3)
S_ = ica.fit_transform(data_df)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
#assert np.allclose(data_df, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(data_df)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

plt.figure()

models = [data_df, S_, H]
names = ['Observations (mixed signal)',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'black', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()