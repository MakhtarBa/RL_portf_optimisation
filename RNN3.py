# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:31:20 2017

@author: Makhtar Ba
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

from sklearn.metrics import mean_squared_error

os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')

data=pd.read_csv('RawData.csv', sep=",")

data=data.scale()
print(data.columns)

df = pd.DataFrame(data['Volume'])
df.index = data['Ntime']

data1 = data.copy()
del data1['Ntime']
data1.drop(['time', 'Close Price','Volume','Low Price', 'High Price'], axis=1, inplace=True)




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


def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

#Initializing the dataset 

train_up=1500
test_up=2000
train_down=0
test_down=1500


return_series=data1['Open Price']
return_df = (return_series - return_series.shift(1))/return_series

data1=pd.concat([data1,pd.DataFrame(return_df)],axis=1)

del data1['Open Price']

training_data=data1.values[train_down:train_up]
testing_data=data1.values[test_down:test_up]

scaler,scaled_train, scaled_test=scale(training_data,testing_data)




number_batch = 200
train_num = 3000
num_nodes = 2 #no. of neurons (makhtar)
num_unrollings =  10
#rolling_window_size = 100
look_back = 20 # number of days to lookback, length of the input time series
#a is inmput data
#a=np.array([testing_data[column_name][j:j+look_back].values for j in range(1, len(return_df[column_name])-look_back)],np.float32)
delta=0.02 #transaction cost (bps)
time=100 #start_time
num_indicators=scaled_train.shape[1]


num_epochs = 3

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
    
'''

   Up to this 
   
'''


#if forced meaning that we put the real signal in the inout of each layer 
with g.as_default():
    #lets define the input variables containers 
    
    reg_tf = tf.constant(0.0) #regularization constant

    input_data = []
    for i in range(num_unrollings):
        input_data.append(tf.placeholder(tf.float32,shape=(num_indicators,look_back),name='input'))

 
    #Variables
    #input matrix
    U = tf.Variable(tf.truncated_normal([num_indicators,look_back,num_nodes],-0.1,0.1))
    #U = tf.Variable(tf.zeros([look_back,num_nodes]))
    
    #recurrent matrix multiplies previous output
    W = tf.Variable(tf.truncated_normal([1,num_nodes],-0.1,0.1))
    #W = tf.Variable(tf.zeros([1,num_nodes]))
    
    #bias vector
    #b = tf.Variable(tf.truncated_normal([1,2*num_nodes],-0.5,0.5))
    b = tf.Variable(tf.zeros([1,2*num_nodes]))
     
    #output matrix wieths after the activation function
    
    #V = tf.Variable(tf.truncated_normal([2*num_nodes,1],-0.1,0.1))
    V = tf.Variable(tf.zeros([2*num_nodes,1]))
    #c = tf.Variable(tf.truncated_normal([1,1],-0.5,0.5))
    c = tf.Variable(tf.zeros([1,1]))
    
    #model
    
    # Recheck the dimensions of the multiplications of matrices accrding to the paper 
    #when training truncate the gradients after num_unrollings
    #print(tf.reshape(tf.sign(input_data[0][0][0]),[1,1]))
    tensor_real_returns=tf.reshape(input_data[num_indicators-1][0][look_back-1],[1,1]) #appending later, tensor shape preserved
    for i in range(num_unrollings):

        
        if i == 0:
            #output_data_feed=tf.reshape(tf.cast(tf.sign(input_data[0][0][look_back-1]), tf.float32),[1,1])        
            output_data_feed=tf.reshape(tf.sign(input_data[num_indicators-1][0][look_back-1]),[1,1])        
            output_ = tf.sign(input_data[num_indicators-1][0][look_back-1])
            for j in range(num_indicators):    
                a_ = tf.concat((tf.matmul(input_data[i],U[j]),output_*W),axis=1)+b
            h_output = tanh(a_)
            output_after= tanh(tf.matmul(h_output,V)+c)
            
        else:
            a_ = tf.concat((tf.matmul(input_data[num_indicators-1][i],U),output_after*W),axis=1)+b
            h_output = tanh(a_)
            output_after= tanh(tf.matmul(h_output,V)+c)
                

        output_data_feed=tf.concat((output_data_feed,output_after), axis=0)
        tensor_real_returns=tf.concat((tensor_real_returns,tf.reshape(input_data[i][0][0],[1,1])),axis=0)
        
    #mean=tf.reduce_mean(output_data)    
    #print(tensor_real_returns,output_data_feed)
    observed_returns=tf.multiply(output_data_feed[0:-1],tensor_real_returns[1:])+delta*tf.abs(tf.subtract(output_data_feed[1:],output_data_feed[0:-1]))
    
    print(observed_returns.shape)
    
    #train
 
    #log likelihood loss
    #global_step = tf.Variable(10)
    global_step = tf.Variable(0,trainable=False)
    L2_loss = tf.nn.l2_loss(U)
    
    #loss = -1*tf.reduce_mean(observed_returns)+reg_tf*L2_loss
    
    mean, variance = tf.nn.moments(observed_returns, axes = [0])
    sharpe = mean/tf.sqrt(variance) 
    R2 = reg_tf*L2_loss
    
    loss = -sharpe
    #-mean # + R2
    
    learning_rate = tf.train.exponential_decay(
        learning_rate=1e-3 ,global_step=global_step, decay_steps=5, decay_rate=0.1, staircase=True)
     
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
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
    np.set_printoptions(precision=3)


    sess=tf.Session() #graph=g)
    
    merged=tf.summary.merge_all()
    
    graph_writer = tf.summary.FileWriter(summaries_dir + '/train',sess.graph)
    
    
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
            #print('U = ', sess.run(U))
            #print('W = ', sess.run(W))
            #print('b = ', sess.run(b))
            #print('V = ', sess.run(V))
            #print('c = ', sess.run(c))
            
            #print('mean of W = ', sess.run(tf.reduce_mean(W)))
            #print('learning rate = ', sess.run(learning_rate))  
            #rint('output = ', sess.run(observed_returns,feed_dict=feed_input))
            
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
            print('gradient_U = ', grad_U_val)
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


