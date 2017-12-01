#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:25:06 2017

@author: admin
"""

import numpy as np
import sklearn
import pandas as pd
from pandas import Series
import os
import datetime
from sklearn.decomposition import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


os.chdir('/Users/admin/Documents/Big data/data')


with open('Tickers.txt') as tickers:
    reader=tickers.read().split("\n")
    list_tickers=[read for read in reader]

#data=pd.read_csv('AAPL.txt'.format(ticker), sep=",")
data=pd.read_csv('AAPL.txt', sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]


data=data[" OPEN"]
data_df=pd.DataFrame(data)

print(type(data_df))
damaged_stocks = []
nondamaged_stocks = ['AAPL']
for ticker in list_tickers[2:]:        
    df = pd.read_csv('{}.txt'.format(ticker.strip()), sep=",")
    df['DATE']=df['DATE'].apply(lambda x : str(x))
    df['DATE']=df['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))
    df.index=df["DATE"]
    df = pd.DataFrame(df[" OPEN"])
    if len(df)<3000:
        #print(ticker)
        damaged_stocks.append(ticker.strip())        
    else :
        #print(type(df))
        data_df=data_df.merge(df,left_index=True,right_index=True)
        nondamaged_stocks.append(ticker.strip())  
    #data=data.merge(data,df)
    
#data_df.columns=list_tickers[1:]
data_df.columns=nondamaged_stocks
data_df=data_df.transpose()

data1 = data_df.copy()
#data1.shape
data1 = data1.iloc[0:1,0:2607]
data1 = data1.transpose()


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
    
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
    
def returns(dataset):
	returns = list()
	for i in range(1, len(dataset)):
		value = (dataset[i]/dataset[i - 1]) - 1
		returns.append(value)
	return Series(returns)
    
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
    
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
    
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
    
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
    
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
    
data1_values = data1.values
returns1 = returns(data1_values)
supervised = timeseries_to_supervised(returns1, 1)
train, test = supervised.values[0:-int(np.floor(0.2*len(supervised)))], supervised.values[-int(np.floor(0.2*len(supervised))):]

# scale the data
# scaler, train_scaled, test_scaled = scale(train, test)

#%%time
# fit the model
lstm_model = fit_lstm(train, 1, 10, 2)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train[:, 0].reshape(len(train), 1, 1)
predicted = lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
expected = list()
for i in range(len(test)):
	# make one-step forecast
	X, y = test[i, 0:-1], test[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# store forecast
	predictions.append(yhat)
	expected.append(y)

rmse = np.sqrt(mean_squared_error(expected, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(expected)
plt.plot(predictions)
plt.show()