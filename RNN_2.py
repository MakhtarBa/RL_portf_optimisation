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
import time as tm

flags = tf.app.flags
FLAGS = flags.FLAGS


#%%
def sigmoid(x):
    return 1/(1+tf.exp(-x))

#%%
os.chdir('/Users/Drogon/Documents/Columbia/Courses/Fall_2017/Big_Data_Choromanski/data/data')

summaries_dir = '/Users/Drogon/Documents/Columbia/Courses/Fall_2017/Big_Data_Choromanski/Codes/Reinforcement_Learning_Sharpe'

#Initializing the dataset 
data=pd.read_csv('GM.txt', sep=",")
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

number_batch = 100
train_num = 3000
num_nodes = 2 #no. of neurons (makhtar)
num_unrollings =  2
#rolling_window_size = 100
look_back = 20 # number of days to lookback, length of the input time series
#a is inmput data
a=np.array([return_df[column_name][j:j+look_back].values for j in range(1, len(return_df[column_name])-look_back)],np.float32)
delta=0.02 #transaction cost (bps)
time=100 #start_time


num_epochs = 1

time1 = tm.time()
    
g = tf.Graph()

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
    
    #U = tf.Variable(tf.truncated_normal([look_back,num_nodes],-0.1,0.1))
    U = tf.Variable(tf.zeros([look_back,num_nodes]))
    
    #recurrent matrix multiplies previous output
    #W = tf.Variable(tf.truncated_normal([1,num_nodes],-0.1,0.1))
    W = tf.Variable(tf.zeros([1,num_nodes]))
    
    #bias vector
    b = tf.Variable(tf.zeros([1,2*num_nodes]))
 
    #output matrix wieths after the activation function
    
    #V = tf.Variable(tf.truncated_normal([2*num_nodes,1],-0.1,0.1))
    V = tf.Variable(tf.zeros([2*num_nodes,1]))
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
    #print(tf.reshape(tf.sign(input_data[0][0][0]),[1,1]))
    tensor_real_returns=tf.reshape(input_data[0][0][look_back-1],[1,1]) #appending later, tensor shape preserved
    for i in range(num_unrollings):
        '''
        if i == 0:
            #output_data_feed=tf.reshape(tf.cast(tf.sign(input_data[0][0][look_back-1]), tf.float32),[1,1])        
            output_data_feed=tf.reshape(tf.sign(input_data[0][0][look_back-1]),[1,1])        
            output_ = tf.sign(input_data[0][0][look_back-1])
            a_ = tf.concat((tf.matmul(input_data[i],U),output_*W),axis=1)+b
            h_output = tf.nn.softmax(a_)
            output_after= tf.nn.softmax(tf.matmul(h_output,V)+c)
            #output_after= 2*tf.cast((output_after+0.5),tf.int32)- 1 #tf.cast(x, tf.int32)
        else:
            a_ = tf.concat((tf.matmul(input_data[i],U),output_after*W),axis=1)+b
            h_output = tf.nn.softmax(a_)
            output_after= tf.nn.softmax(tf.matmul(h_output,V)+c)    
            #output_after= 2*tf.cast((output_after+0.5),tf.int32)- 1 #tf.cast(x, tf.int32)
        '''
        
        if i == 0:
            #output_data_feed=tf.reshape(tf.cast(tf.sign(input_data[0][0][look_back-1]), tf.float32),[1,1])        
            output_data_feed=tf.reshape(tf.sign(input_data[0][0][look_back-1]),[1,1])        
            output_ = tf.sign(input_data[0][0][look_back-1])
            a_ = tf.concat((tf.matmul(input_data[i],U),output_*W),axis=1)+b
            h_output = sigmoid(a_)
            output_after= sigmoid(tf.matmul(h_output,V)+c)
            #output_after= 2*tf.cast((output_after+0.5),tf.int32)- 1 #tf.cast(x, tf.int32)
        else:
            a_ = tf.concat((tf.matmul(input_data[i],U),output_after*W),axis=1)+b
            h_output = sigmoid(a_)
            output_after= sigmoid(tf.matmul(h_output,V)+c)
            #output_after= 2*tf.cast((output_after+0.5),tf.int32)- 1 #tf.cast(x, tf.int32)
                
            
        #print(output_after.dtype,output_data_feed.dtype)
        signal= 2*tf.floor((output_after+0.5))- 1
        output_data_feed=tf.concat((output_data_feed,signal), axis=0)
        tensor_real_returns=tf.concat((tensor_real_returns,tf.reshape(input_data[i][0][0],[1,1])),axis=0)
    
    #print(tensor_real_returns,output_data_feed)
    observed_returns=tf.multiply(output_data_feed[0:-1],tensor_real_returns[1:])+delta*tf.abs(tf.subtract(output_data_feed[1:],output_data_feed[0:-1]))
    
    print(observed_returns.shape)
    
    #train
 
    #log likelihood loss
    #global_step = tf.Variable(10)
    global_step = tf.constant(10)
    L2_loss = tf.nn.l2_loss(U)
    
    #loss = -1*tf.reduce_mean(observed_returns)+reg_tf*L2_loss
    
    mean, variance = tf.nn.moments(observed_returns, axes = [0])
    sharpe = mean/tf.sqrt(variance) 
    R2 = reg_tf*L2_loss
    
    loss = tf.reduce_sum(observed_returns)
    #-mean # + R2
    
    learning_rate = tf.train.exponential_decay(
        learning_rate=10e1 ,global_step=global_step, decay_steps=5, decay_rate=0.1, staircase=True)
     
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    #grad_compute =optimizer.compute_gradients(loss) #,var_list=[U,W,b,V,c])
    #gradients_clipped, _ = tf.clip_by_global_norm(grad_operations, 1.25)
    
    #grad_apply=optimizer.apply_gradients(grad_compute)
    #grad=tf.gradients(loss,[U,W,b,V,c])

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
    np.set_printoptions(precision=3)


    sess=tf.Session() #graph=g)
    
    merged=tf.summary.merge_all()
    
    graph_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
    
    
    sess.run(init2)
    
    global_loss_value = []
    global_sharpe_value = []
    global_last_signal_value = []    
    global_last_output_value = []        
    for epoch_i in range((num_epochs)):
        
        loss_value=[]
        sharpe_value = []
        last_signal_value = []
        last_output_value = []  
        epoch_time = time
    
        for j in range(number_batch): 
            #global_step=global_step+1
            epoch_time=epoch_time+1
            feed_input={input_data[i]:a[epoch_time-num_unrollings+i+j].reshape(1,look_back) for i in range(num_unrollings)}
            #grad_vals=sess.run(grad,feed_dict=feed_input)
            
            
            #print('input data = ', feed_input)
            print('U = ', sess.run(U))
            print('W = ', sess.run(W))
            print('b = ', sess.run(b))
            print('V = ', sess.run(V))
            print('c = ', sess.run(c))
            
            print('mean of W = ', sess.run(tf.reduce_mean(W)))
            print('learning rate = ', sess.run(learning_rate))  
            print('output = ', sess.run(observed_returns,feed_dict=feed_input))
            
            #grad_, loss_tf_val, temp_sharpe, last_signal, last_output= sess.run([grad_calcul, loss, sharpe, output_data_feed[num_unrollings - 1],output_after[0][0]], feed_dict=feed_input)     
            temp_sharpe = sess.run(sharpe,feed_dict=feed_input)
            loss_tf_val = sess.run(loss,feed_dict=feed_input)
            
            print('Loss = ', loss_tf_val)
            
            last_signal = sess.run(output_data_feed[num_unrollings - 1],feed_dict=feed_input)
            last_output = sess.run(output_after[0][0],feed_dict=feed_input)
            summary=sess.run(merged,feed_dict=feed_input)
            #garb, grad_ =sess.run([optimizer, grad],  feed_dict=feed_input)
            #grad_=sess.run([grad_apply],  feed_dict=feed_input)
            #grad_ = sess.run(grad_operations, feed_dict=feed_input)
            sess.run(optimizer, feed_dict=feed_input)
            
            loss_value.append(loss_tf_val)
            sharpe_value.append(temp_sharpe)
            last_signal_value.append(last_signal)
            last_output_value.append(last_output)
            #print(grad_[0])
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
temp_vals = []
#%%

i = num_epochs - 1
#%%
i = 1- 1
#%%
clean_sharpe_value = [sharpe_value_temp for sharpe_value_temp in global_sharpe_value[i] if sharpe_value_temp <= 2]
dirty_sharpe_value = global_sharpe_value[i]

graph_list = make_list(dirty_sharpe_value)


temp_vals.append(global_last_signal_value[i])

'''temp_vals = temp_vals[0]
temp_vals = (make_list(temp_vals))
'''

#%%



plt.plot(global_last_output_value[i])

#%%

sharpe_mean = [np.mean(sharpe_series) for sharpe_series in global_sharpe_value]
sharpe_std = [np.std(sharpe_series) for sharpe_series in global_sharpe_value]


