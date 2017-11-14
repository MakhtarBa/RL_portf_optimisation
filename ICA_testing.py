# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 02:05:43 2017

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



os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')

with open('Tickers.txt') as tickers:
    reader=tickers.read().split("\n")
    list_tickers=[read for read in reader]

#Initializing the dataset 
data=pd.read_csv('AAPL.txt', sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]


data=data[" OPEN"]
data_df=pd.DataFrame(data)

#Loading the files 
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

list_tickers = [x for x in list_tickers if x  not in damaged_stocks]
data_df.columns=list_tickers[1:]
data_df=data_df.transpose()
# replicate Data from question in DataFrame


#Test if there is no big absence of data 
dates=data_df.columns
lag=[(dates[i]-dates[i+1]).days for i in range (len(dates)-1)]
lag_dataframe = pd.DataFrame(lag)
lag_dataframe.plot(kind='hist')

return_df=pd.DataFrame()

for i in range(len(list(dates))-1):
    return_df[dates[i]]=data_df[dates[i+1]]/data_df[dates[i]]-1


##########################################


'''

Setting up the ICA 

'''

##########################################
return_df=return_df.transpose()
ica = FastICA(n_components=return_df.shape[1])
Ind_components = ica.fit_transform(return_df)  # Reconstruct independ constituents 
mixing = ica.mixing_  # Get estimated mixing matrix
unmixing=np.linalg.inv(mixing)

ica_mean=np.array(ica.mean_.transpose()).reshape(1,len(ica.mean_.transpose()))

ica_mean=np.tile(ica_mean,(len(return_df),1)) # put then mean in the correct format 

test_Ind_components=Ind_components+(ica_mean).dot(unmixing)

test_Ind_components=pd.DataFrame(test_Ind_components,columns=return_df.columns)
plt.plot(test_Ind_components['AAPL'],color='orange')
plt.plot(test_Ind_components['ABT'],color='orange')
test_Ind_components.describe()
test_Ind_components.to_csv('meaned_Ind_components.csv')
 
reconstitution_ICA=test_Ind_components.dot(mixing)
reconstitution_ICA.columns=return_df.columns
# For comparison, compute PCA
pca = PCA(n_components=92)
H = pca.fit_transform(return_df)  # Reconstruct signals based on orthogonal components
reconstitution_PCA= pca.inverse_transform(H)  # Reconstruct signals based on orthogonal components
reconstitution_PCA=pd.DataFrame(reconstitution_PCA)

reconstitution_PCA.columns=return_df.columns

    
# #############################################################################
# Plot results


# Comparison of the sumcum 

cumsum_ICA=pd.DataFrame(columns=reconstitution_ICA.columns)
cumsum_PCA=pd.DataFrame(columns=reconstitution_ICA.columns)
cumsum_return=pd.DataFrame(columns=reconstitution_ICA.columns)


for column in reconstitution_ICA.columns:
    cumsum_ICA[column]=reconstitution_ICA[column].cumsum()
    cumsum_PCA[column]=reconstitution_PCA[column].cumsum()
    cumsum_return[column]=return_df[column].cumsum()

fig=plt.figure()

plt.plot(cumsum_ICA['AAPL'],color='orange')
plt.plot(cumsum_PCA['AAPL'],color='green')
plt.plot(cumsum_return['AAPL'],color='black')

plt.show()

