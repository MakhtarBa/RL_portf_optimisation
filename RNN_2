#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:41:41 2017

@author: Drogon
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
import tensorflow as tf

#%%
os.chdir('/Users/Drogon/Documents/Columbia/Courses/Fall_2017/Big_Data_Choromanski/data/data')

#Initializing the dataset 
data=pd.read_csv('AAPL.txt', sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]

data=data[" OPEN"]
data_df=pd.DataFrame(data)

#%%
return_df = (data_df - data_df.shift(1))/data_df
#%%




#%%
column_name = ' OPEN'

number_batch = 500
num_nodes = 2
num_unrollings =  10
rolling_window_size = 100
look_back = 20 # number of days to lookback, length of the input time series
a=np.array([return_df[column_name][j:j+look_back].values for j in range(len(return_df[column_name])-look_back)],np.float32)
delta=0.2
time=100


num_epochs = 500
    
g = tf.Graph()

#if forced meaning that we put the real signal in the inout of each layer 
with g.as_default():
    #lets define the input variables containers 
    
    reg_tf = tf.constant(0.01)

    input_data = []
    for i in range(num_unrollings):
        input_data.append(tf.placeholder(tf.float32,shape=(None,look_back),name='input'))
    
      #the actual signals 
    #output_data_feed= tf.placeholder(tf.float32,shape=(rolling_window_size-num_unrollings,None),name='one') #the minus one is to say that we input the real signal only on the previous days 
    
    #Validation returns 
    #tensor_real_returns=tf.placeholder(tf.float32,shape=(1,rolling_window_size-num_unrollings))
    

 
    #Variables
    #input matrix
    
    U = tf.Variable(tf.truncated_normal([look_back,num_nodes],-0.1,0.1))
    
    #recurrent matrix multiplies previous output
    W = tf.Variable(tf.truncated_normal([1,num_nodes],-0.1,0.1))
 
    #bias vector
    b = tf.Variable(tf.zeros([1,2*num_nodes]))
 
    #output matrix wieths after the activation function
    
    V = tf.Variable(tf.truncated_normal([2*num_nodes,1],-0.1,0.1))
    c = tf.Variable(tf.zeros([1,1]))
 
    #model
    '''
    def RNN(input_,o_input):
        global W
        
        a = tf.concat(1,[tf.matmul(input_,U),o_input*W)])+b
        h_output = tf.nn.tanh(a)
        o_out = tf.matmul(h_output,V)+c
        
        return o_out
    
    # Recheck the dimensions of the multiplications of matrices accrding to the paper 
    '''
    #when training truncate the gradients after num_unrollings
    print(tf.reshape(tf.sign(input_data[0][0][0]),[1,1]))
    tensor_real_returns=tf.reshape(input_data[0][0][0],[1,1])
    for i in range(num_unrollings):
        
        if i == 0:
            output_data_feed=tf.reshape(tf.cast(tf.sign(input_data[0][0][0]), tf.float32),[1,1])        
            output_ = tf.sign(input_data[0][0][look_back-1])
            a_ = tf.concat((tf.matmul(input_data[i],U),output_*W),axis=1)+b
            h_output = tf.nn.softmax(a_)
            output_after= tf.matmul(h_output,V)+c    
            
        else:
            
            a_ = tf.concat((tf.matmul(input_data[i],U),output_after*W),axis=1)+b
            h_output = tf.nn.softmax(a_)
            output_after= tf.matmul(h_output,V)+c    
        #print(output_after.dtype,output_data_feed.dtype)
        output_data_feed=tf.concat((output_data_feed,output_after), axis=0)
        tensor_real_returns=tf.concat((tensor_real_returns,tf.reshape(input_data[i][0][0],[1,1])),axis=0)
    
    #print(tensor_real_returns,output_data_feed)
    observed_returns=np.multiply(output_data_feed[1:],tensor_real_returns[0:num_unrollings])+delta*np.abs(np.array(output_data_feed[1:])-np.array(output_data_feed[0:-1]))
    print(observed_returns.shape)
    
    #train
 
    #log likelihood loss
    global_step = tf.Variable(0)
    L2_loss = tf.nn.l2_loss(U)
    
    #loss = -1*tf.reduce_mean(observed_returns)+reg_tf*L2_loss
    
    mean, variance = tf.nn.moments(observed_returns, axes = [0])
    sharpe = mean/tf.sqrt(variance) 
    R2 = reg_tf*L2_loss
    
    loss = -sharpe + R2
    
    learning_rate = tf.train.exponential_decay(
        learning_rate=2.5,global_step=global_step, decay_steps=5000, decay_rate=0.1, staircase=True)
     
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    grad_operations=optimizer.compute_gradients(loss,var_list=[U,W,b,V,c])
    gradients_clipped, _ = tf.clip_by_global_norm(grad_operations, 1.25)
    
    grad_calcul=optimizer.apply_gradients(grad_operations)
    grad=tf.gradients(loss,[U,W,b,V,c])

    
    
    #optimizer
    '''
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients,var=zip(*optimizer.compute_gradients(loss))
    opt=optimizer.apply_gradients(zip(gradients_clipped,var),global_step=global_step)
    '''
    
    init = tf.global_variables_initializer()
    np.set_printoptions(precision=3)

     
    sess=tf.Session(graph=g)
    sess.run(init)

    global_loss_value = []
    global_sharpe_value = []
        
    for epoch_i in range((num_epochs)):
        
        loss_value=[]
        sharpe_value = []
        
        epoch_time = time
    
        for j in range(number_batch):            
            epoch_time=epoch_time+1
            feed_input={input_data[i]:a[epoch_time-num_unrollings+i].reshape(1,look_back) for i in range(num_unrollings)}
            #grad_vals=sess.run(grad,feed_dict=feed_input)
    
            grad_, loss_tf_val, temp_sharpe = sess.run([grad_calcul, loss, sharpe], feed_dict=feed_input)     
            loss_value.append(loss_tf_val)
            sharpe_value.append(temp_sharpe)           
            #print(grad_vals)
            #print(grad_)
            #update=sess.run(tf.norm(grad_vals[0]*learning_rate))
            '''
            count=0     
            
            while(count<10):
                count+=1
                grad=tf.gradients(loss,[U,W,b,V,c])
    
                grad_operations_val , loss_tf_val, grad_calcul_val = sess.run([grad_operations, loss, grad_calcul], feed_dict=feed_input)
                grad_val=sess.run(grad, feed_dict=feed_input)
    
                loss_value.append(loss_tf_val)
                update=sess.run(tf.norm(grad_val[0]*learning_rate))
            '''    
                  
        global_loss_value.append(loss_value)
        global_sharpe_value.append(sharpe_value)
        
#%%

i = 20 - 1

clean_sharpe_value = [sharpe_value_temp for sharpe_value_temp in global_sharpe_value[i] if sharpe_value_temp <= 2]
dirty_sharpe_value = global_sharpe_value[i]

#%%

sharpe_mean = [np.mean(sharpe_series) for sharpe_series in global_sharpe_value]
sharpe_std = [np.std(sharpe_series) for sharpe_series in global_sharpe_value]
