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
import tensorflow as tf
import time as tm
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

os.chdir('/Users/Drogon/Documents/Columbia/Courses/Fall_2017/Big_Data_Choromanski/data/data')
summaries_dir = '/Users/Drogon/Documents/Columbia/Courses/Fall_2017/Big_Data_Choromanski/Codes/Reinforcement_Learning_Sharpe'

data=pd.read_csv('RawData.csv', sep=",")

print(data.columns)
#%%
df = pd.DataFrame(data['Volume'])
df.index = data['Ntime']

data1 = data.copy()
del data1['Ntime']
data1.drop(['time', 'Close Price','Volume','Low Price', 'High Price'], axis=1, inplace=True)


#%%

rf_rate = data1['Federal Fund Rate'].copy()

rf_rate/=36500 ###rf_rate is daily adjusted for percentage

#%%
#Initializing the dataset 

data_df=data1['Open Price'].copy()

#%%
return_df = (data_df - data_df.shift(1))/data_df
return_df = pd.DataFrame(return_df)

#%%

plt.plot(return_df.cumsum())


#%%
column_name = 'Open Price'

number_batch = 100 #1500
num_nodes = 2 #no. of neurons (makhtar)
num_unrollings =  1
#rolling_window_size = 100
look_back = 20 # number of days to lookback, length of the input time series
#a is inmput data
a=np.array([return_df[column_name][j:j+look_back].values for j in range(1, len(return_df[column_name])-look_back)],np.float32)
delta=0.0002 #transaction cost (bps)
time=25#start_time
#%%

num_epochs = 1

time1 = tm.time()
    
g = tf.Graph()

#%%

def learning_rate_exponential(rate0, gamma, global_, decay):
    return rate0*gamma**(global_/decay)

def modified_learning_rate_exponential(rate0, ratef, number_batch, global_):
    gamma = ratef/rate0
    return rate0 * gamma**((global_+1)/number_batch)

def constant_learning_rate(rate0):
    return rate0

def sigmoid(x):
    return 1/(1+tf.exp(-x))
def relu_modified(x):
    return (tf.exp(x)-1)

def sigmoid_modified(w,center,b_):
    return (tf.exp(b_*((w-center)/center))/(1+tf.exp(b_*(w-center)/center))-center)/(1/(1+tf.exp(-b_))-center)

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def tanh(x):
  #return (tf.exp(x)-tf.exp(-x))/(tf.exp(x)+tf.exp(-x))
   return 2*sigmoid(2*x)-1 



#%%
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    '''tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)'''
   
def test_signal(start_date, end_date,column_name,epoch):
    
    
    global return_df
    global global_last_signal_value
    
    
    signal_convergence=[]
    for t in range(start_date, end_date):
        return_component=return_df[column_name][t+1]
        
        if return_component*global_last_signal_value[epoch][t-start_date][0]>0:
            signal_convergence.append(1)
        else :
            signal_convergence.append(0)
        
    return np.mean(signal_convergence)



#if forced meaning that we put the real signal in the inout of each layer 
with g.as_default():
    #lets define the input variables containers 
    
    reg_tf = tf.constant(0.0) #regularization constant

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
    #U = tf.Variable(tf.zeros([look_back,num_nodes]))
    '''U = tf.Variable(tf.random_uniform(shape = [look_back,num_nodes],minval=-1, maxval=1,
    dtype=tf.float32))'''
    
    
    
    #recurrent matrix multiplies previous output
    W = tf.Variable(tf.truncated_normal([1,num_nodes],-0.1,0.1))
    #W = tf.Variable(tf.zeros([1,num_nodes]))
    
    
    
    #bias vector
    #b = tf.Variable(tf.truncated_normal([1,2*num_nodes],-0.5,0.5))
    b = tf.Variable(tf.zeros([1,2*num_nodes]))
     
    #output matrix wieths after the activation function
    
    V = tf.Variable(tf.truncated_normal([2*num_nodes,1],-0.1,0.1))
    #V = tf.Variable(tf.zeros([2*num_nodes,1]))
    #c = tf.Variable(tf.truncated_normal([1,1],-0.5,0.5))
    #c = tf.Variable(tf.zeros([1,1]))
    c = tf.Variable([0.0], [1,1])
    
    #model
    
    # Recheck the dimensions of the multiplications of matrices accrding to the paper 
    #when training truncate the gradients after num_unrollings
    #print(tf.reshape(tf.sign(input_data[0][0][0]),[1,1]))
    tensor_real_returns=tf.reshape(input_data[0][0][look_back-1],[1,1]) #appending later, tensor shape preserved
    for i in range(num_unrollings):

        
        if i == 0:
            #output_data_feed=tf.reshape(tf.cast(tf.sign(input_data[0][0][look_back-1]), tf.float32),[1,1])        
            output_data_feed=tf.reshape(tf.sign(input_data[0][0][look_back-1]),[1,1])        
            output_ = tf.sign(input_data[0][0][look_back-1])
            a_ = tf.concat((tf.matmul(input_data[i],U),output_*W),axis=1)+b
            h_output = tanh(a_)
            output_after= tanh(tf.matmul(h_output,V)+c)
            #output_after= tf.matmul(h_output,V)+c
            #output_after= 2*tf.cast((output_after+0.5),tf.int32)- 1 #tf.cast(x, tf.int32)
        else:
            a_ = tf.concat((tf.matmul(input_data[i],U),output_after*W),axis=1)+b
            h_output = tanh(a_)
            output_after= tanh(tf.matmul(h_output,V)+c)
            #output_after= tf.matmul(h_output,V)+c
            #output_after= 2*tf.cast((output_after+0.5),tf.int32)- 1 #tf.cast(x, tf.int32)
                

        #signal= sigmoid_modified(output_after,0.5,100.0)
        output_data_feed=tf.concat((output_data_feed,output_after), axis=0)
        tensor_real_returns=tf.concat((tensor_real_returns,tf.reshape(input_data[i][0][look_back-1],[1,1])),axis=0)
        
    #mean=tf.reduce_mean(output_data)    
    #print(tensor_real_returns,output_data_feed)
    observed_returns=tf.multiply(output_data_feed[0:-1],tensor_real_returns[1:])+delta*tf.abs(tf.subtract(output_data_feed[1:],output_data_feed[0:-1]))
    
    print(observed_returns.shape)
    
    #train
 
    #log likelihood loss
    #global_step = tf.Variable(10)
    #global_step = tf.Variable(0,trainable=False)
    L2_loss = tf.nn.l2_loss(U)
    
    #loss = -1*tf.reduce_mean(observed_returns)+reg_tf*L2_loss
    
    mean, variance = tf.nn.moments(observed_returns, axes = [0])
    sharpe = mean/tf.sqrt(variance) 
    R2 = reg_tf*L2_loss
    
    loss = -mean
    #-mean # + R2
    
    '''learning_rate = tf.train.exponential_decay(
        learning_rate=5e-2 ,global_step=global_step, decay_steps=5, decay_rate=0.01, staircase=True)'''
    
    var_learning_rate = tf.Variable(0.0,trainable=False)
    
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    optimizer=tf.train.GradientDescentOptimizer(var_learning_rate)
    grad_compute =optimizer.compute_gradients(loss) #,var_list=[U,W,b,V,c])
    #gradients_clipped, _ = tf.clip_by_global_norm(grad_operations, 1.25)
    
    grad_apply=optimizer.apply_gradients(grad_compute)
    grad_U=tf.gradients(loss,[U])
    grad_V=tf.gradients(loss,[V])
    grad_W=tf.gradients(loss,[W])
    grad_b=tf.gradients(loss,[b])
    grad_c=tf.gradients(loss,[c])
    

    #variable_summaries(W)
    W_mean = tf.reduce_mean(W)
    tf.summary.scalar('mean', W_mean)
    
    #optimizer
    '''
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients,var=zip(*optimizer.compute_gradients(loss))
    opt=optimizer.apply_gradients(zip(gradients_clipped,var),global_step=global_step)
    '''
    
    init2 = tf.global_variables_initializer()     
    np.set_printoptions(precision=6)


    sess=tf.Session() #graph=g)
    
    merged=tf.summary.merge_all()
    
    graph_writer = tf.summary.FileWriter(summaries_dir + '/train',sess.graph)
    
    
    sess.run(init2)
    
    global_ = 0
    
    global_loss_value = []
    global_sharpe_value = []
    global_last_signal_value = []    
    global_last_output_value = []        
    for epoch_i in range((num_epochs)):
        #global_step = tf.Variable(0,trainable=False)
        loss_value=[]
        sharpe_value = []
        last_signal_value = []
        last_output_value = []  
        epoch_time = time
    
        for j in range(number_batch): 
            
            ###
            '''rate0 = 5e-2 
            gamma = 0.9
            decay = 5.
            global_ += 1'''
            
            rate0 = 5e3 
            ratef = 1e-3
            global_ += 1
            for sub_batch in range(10):

                ###
                #temp_learning_rate = learning_rate_exponential(rate0, gamma, global_, decay)
                #temp_learning_rate = modified_learning_rate_exponential(rate0, ratef, number_batch, global_)
                temp_learning_rate = constant_learning_rate(rate0)
                
                temp_learning_rate = tf.cast(temp_learning_rate, dtype = tf.float32)
                var_learning_rate = temp_learning_rate 
                #learning_rate_dict = {var_learning_rate: temp_learning_rate}
                
                #global_step=global_step+1
                epoch_time=time+j
                feed_input={input_data[i]:a[epoch_time-num_unrollings+i].reshape(1,look_back) for i in range(num_unrollings)}
                #feed_input.update(learning_rate_dict)
    
                #grad_vals=sess.run(grad,feed_dict=feed_input)
                
                
                #print('input data = ', feed_input)
                #print('U = ', sess.run(U))
                #print('W = ', sess.run(W))
                #print('b = ', sess.run(b))
                #print('V = ', sess.run(V))
                #print('c = ', sess.run(c))
                
                #print('mean of W = ', sess.run(tf.reduce_mean(W)))
                print('learning rate = ', sess.run(var_learning_rate))#, feed_dict = learning_rate_dict))  
                #print('output = ', sess.run(observed_returns,feed_dict=feed_input))
                
                #grad_, loss_tf_val, temp_sharpe, last_signal, last_output= sess.run([grad_calcul, loss, sharpe, output_data_feed[num_unrollings - 1],output_after[0][0]], feed_dict=feed_input)     
                temp_sharpe = sess.run(sharpe,feed_dict=feed_input)
                loss_tf_val = sess.run(loss,feed_dict=feed_input)
                
                print('Loss = ', loss_tf_val)
                
                last_signal = sess.run(output_data_feed[num_unrollings - 1],feed_dict=feed_input)
                last_output = sess.run(output_after[0][0],feed_dict=feed_input)
                summary=sess.run(merged,feed_dict=feed_input)
                garb =sess.run(grad_compute,  feed_dict=feed_input)
                
                grad_=sess.run([grad_apply],  feed_dict=feed_input)
                grad_c_val=sess.run(grad_c,  feed_dict=feed_input)
                grad_b_val=sess.run(grad_b,  feed_dict=feed_input)
                grad_U_val=sess.run(grad_U,  feed_dict=feed_input)
                grad_V_val=sess.run(grad_V,  feed_dict=feed_input)
                grad_W_val=sess.run(grad_W,  feed_dict=feed_input)
                
                #grad_ = sess.run(grad_operations, feed_dict=feed_input)
                #sess.run(optimizer, feed_dict=feed_input)
                #print('gradient_c = ', grad_c_val)
                #print('gradient_U = ', grad_U_val)
                print('gradient_V = ', grad_V_val)
                #print('gradient_W = ', grad_W_val)
                #print('gradient_b = ', grad_b_val)
                
                
                
                loss_value.append(loss_tf_val)
                sharpe_value.append(temp_sharpe)
                last_signal_value.append(last_signal)
                last_output_value.append(last_output)
                graph_writer.add_summary(summary,j)
                
        global_loss_value.append(loss_value)
        global_sharpe_value.append(sharpe_value)
        global_last_signal_value.append(last_signal_value)
        global_last_output_value.append(last_output_value)
time2 = tm.time()     

#%%
   
print(time2 - time1)

#%%

def make_list(input):
    temp = [input[i][0] for i in range(len(input))]
    return temp

#%%
dirty_sharpe_value_dict = {}
#%%

i = num_epochs - 1
#%%
i = 1- 1

#%%
i = 0

#%% dirty sharpe value
clean_sharpe_value = [sharpe_value_temp[0] for sharpe_value_temp in global_sharpe_value[i] if sharpe_value_temp <= 2]


dirty_sharpe_value = global_sharpe_value[i]

dirty_sharpe_value = [dirty_sharpe_value[i][0] for i in range(len(dirty_sharpe_value))]
plt.plot(dirty_sharpe_value)

#%% last signal

dirty_last_signal_value = []
for j in range(num_epochs):
    dirty_last_signal_value.append(global_last_signal_value[j])

dirty_last_signal_value = np.array(dirty_last_signal_value).flatten().tolist()
plt.plot(dirty_last_signal_value)

#%% loss

dirty_loss_value = []
for j in range(num_epochs):
    dirty_loss_value.append(global_loss_value[j])

dirty_loss_value = np.array(dirty_loss_value).flatten().tolist()
plt.plot(dirty_loss_value)


#%%
##learning rate
plt.plot([modified_learning_rate_exponential(rate0, ratef, number_batch, i) for i in range(number_batch)])


#%%
#plt.plot([global_last_signal_value[i][j][0] for j in range(num)])

#%%

sharpe_mean = [np.mean(sharpe_series) for sharpe_series in global_sharpe_value]
sharpe_std = [np.std(sharpe_series) for sharpe_series in global_sharpe_value]
