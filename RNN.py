

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


os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')

return_df=pd.read_csv('subset_return_df.csv')



batch_size = 10 
num_nodes = 2
num_unrollings = 20
rolling_window_size=100
look_back=3# number of days to lookback, length of the input time series
a=np.array([return_df['AAPL'][j:j+look_back].values for j in range(len(return_df['AAPL'])-look_back)],np.float32)
 
g = tf.Graph()
#if forced meaning that we put the real signal in the inout of each layer 
with g.as_default():
    #lets define the input variables containers 
    
    reg_tf = tf.constant(0.01)

    input_data = []
    for i in range(num_unrollings):
        input_data.append(tf.placeholder(tf.float32,shape=(1,look_back)))
 
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
    
    output_after=0
    for i in range(num_unrollings):
        
        if i == 0:
            
            output_ = tf.sign(input_data[0][0][look_back-1])
            a_ = tf.concat((tf.matmul(input_data[i],U),output_*W),axis=0)+b
            h_output = tf.nn.softmax(a_)
            output_after= tf.matmul(h_output,V)+c    
                        
        else:
            
            a_ = tf.concat([tf.matmul(input_data[i],U),output_after*W])+b
            h_output = tf.nn.softmax(a_)
            output_after= tf.matmul(h_output,V)+c    
            
        output_data_feed=tf.concat(output_data_feed,output_after)
        
    tensor_real_returns=tf.cast([train[i][0] for i in range(num_unrollings)], tf.float32)
    #tensor_real_returns=tf.concat(tensor_real_returns,tensor_real_returns_unrollings)
    
    observed_returns=tf.multiply(output_data_feed,tensor_real_returns)+delta*tf.abs(tf.add(output_data_feed[1:],-1*output_data_feed[0:output_data_feed.shape[0]-1]))
    
    #train
 
    #log likelihood loss
    L2_loss = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(observed_returns)+reg_tf*L2_loss
    learning_rate = tf.train.exponential_decay(
        learning_rate=2.5,global_step=global_step, decay_steps=5000, decay_rate=0.1, staircase=True)
     
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    grad_operations=optimizer.compute_gradients(loss,var_list=[U,W,b,V,c])
    gradients_clipped, _ = tf.clip_by_global_norm(gradients, 1.25)
    
    grad_calcul=optimizer.apply_gradients(gradients_clipped)
    grad=tf.gradients(loss,[U,W,b,V,c])

    
    
    #optimizer
    '''
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients,var=zip(*optimizer.compute_gradients(loss))
    opt=optimizer.apply_gradients(zip(gradients_clipped,var),global_step=global_step)
    '''
    time=1000
    for j in range(batch_size):            
        time=time+j
        feed_input={input_data[i]:a[time-num_unrollings+i] for i in range(num_unrollings)}
        grad_vals=sess.run(grad,feed_dict=feed_input)
        grad_=sess.run(grad_calcul, feed_dict=feed_input)     
        print(grad_vals)
        print(grad_)
        update=sess.run(tf.norm(grad_vals[0]*learning_rate))
        
        count=0     
        loss_value=[]
        
        while(count<10):
            count+=1
            grad=tf.gradients(loss,W1_tf)
            '''
            input_1=tf.reshape(tf.concat((a[0],[1]),axis=0),[1,4])
            input_2=tf.reshape(tf.concat((a[1],[1]),axis=0),[1,4]) 
            input_3=tf.reshape(tf.concat((a[2],[1]),axis=0),[1,4])
            '''
            grad_operations_val , loss_tf_val, grad_calcul_val = sess.run([grad_operations, loss, grad_calcul], feed_dict=feed_input)
            grad_val=sess.run(grad, feed_dict=feed_input)
            
            update=sess.run(tf.norm(grad_val[0]*learning_rate))
            W2_tf=tf.add(W2_tf,grad_val[0]*0)
            loss_value.append(loss_tf_val)
            print(count)
            



    '''
    # Validation
    val_output_after = tf.nn.softmax(RNN(val_input,val_output))
    val_probs = tf.nn.softmax(val_output_after)
    '''
    #add init op to the graph
    #init = tf.initialize_all_variables()
    
    '''
num_steps=50001
b = Batch(text,batch_size)
 
sess=tf.Session(graph=g)
sess.run(init)
average_loss = 0
 
for step in range(num_steps):
    if (step*b.batch_size)%(b.text_size) < b.batch_size:
        print " "
        print "NEW EPOCH"
        print " "
 
    #initialize the output
    if step == 0: #initialize the output state vectors
        output_pass = np.zeros([batch_size,alphabet_size],dtype=np.float32)
    feed_dict={output_feed: output_pass}
 
    #get the new inputs and labels
    batch_x,batch_y=getWindowBatch(b,num_unrollings)
 
    #mega batches
    mega_batch_x = [] #each elemnt will be a batch.  there will be tau elements where tau is the number of unrollings
    mega_batch_y = []
    for n in range(num_unrollings):
        batchx = np.ndarray((batch_size,alphabet_size)) #contain all the one-hot encoding of the characters
        batchy = np.ndarray((batch_size,alphabet_size))
        for ba in range(batch_size):
            batchx[ba]=char2vec(batch_x[n][ba])
            batchy[ba]=char2vec(batch_y[n][ba])
        mega_batch_x.append(batch)
        mega_batch_y.append(batch)
 
    for i in range(num_unrollings):
        feed_dict[train[i]] = mega_batch_x[i]
        feed_dict[labels[i]] = mega_batch_y[i]
 
    output_pass,l,_=sess.run([output_after,loss,opt],feed_dict=feed_dict)
    average_loss += l
    if step % 1000 == 0:
        print 'Average loss: ',str(average_loss/1000)
        average_loss = 0
 
        print 'Learning rate: ', str(learning_rate.eval(session=sess))
        #sample and then generate text
        s=''
 
        #initialize the validations out and character
        val_output_O = vec2mat(char2vec(id2char(sample_prob(random_dist()))))
 
        char_id = sample_prob(random_dist()) #create a random distribution then sample
        val_input_O = vec2mat(char2vec(id2char(char_id)))
 
        s+=id2char(char_id)
        for _ in range(100):
            feed_dict = {val_input: val_input_O,
                         val_output: val_output_O}
            val_output_O,dist = sess.run([val_output_after,val_probs],feed_dict=feed_dict)
            char_id=sample_prob(dist[0])
            val_input_O = vec2mat(char2vec(id2char(char_id)))
            s+=id2char(char_id)
        print s
        



with g.as_default():
    #lets define the input variables containers 
    input_data = []
    for i in range(num_unrollings):
        input_data.append(tf.placeholder(tf.float32,shape=(rolling_window_size,look_back)))
 
    #the previous output the gets fed into the cell
    output_feed= tf.placeholder(tf.float32,shape=(batch_size,alphabet_size),name='one')
 
    #one-hot encoded labels for training
    labels = list()
    for i in range(num_unrollings):
        labels.append(tf.placeholder(tf.float32,shape=(batch_size,alphabet_size)))
 
    #validation place holder
    val_input = tf.placeholder(tf.float32,shape=(1,alphabet_size))
    val_output = tf.placeholder(tf.float32,shape=(1,alphabet_size))
 
    #Variables
    #input matrix
    U = tf.Variable(tf.truncated_normal([alphabet_size,num_nodes],-0.1,0.1))
    
    #recurrent matrix multiplies previous output
    W = tf.Variable(tf.truncated_normal([alphabet_size,num_nodes],-0.1,0.1))
 
    #bias vector
    b = tf.Variable(tf.zeros([1,2*num_nodes]))
 
    #output matrix
    V = tf.Variable(tf.truncated_normal([num_nodes,alphabet_size],-0.1,0.1))
    c = tf.Variable(tf.zeros([1,alphabet_size]))
 
    #model
    def RNN(i,o_input):
        a = tf.concat(1,[tf.matmul(i,U),tf.matmul(o_input,W)])+b
        h_output = tf.nn.tanh(a)
        o_out = tf.matmul(h_output,V)+c
        return o_out
 
    #when training truncate the gradients after num_unrollings
    for i in range(num_unrollings):
        if i == 0:
            outputs = list()
            output_after = RNN(train[i],output_feed)
        else:
            output_after = RNN(train[i],labels[i-1])
        outputs.append(output_after)
 
    #train
 
    #log likelihood loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.concat(0,outputs),tf.concat(0,labels)))
 
    #optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        learning_rate=2.5,global_step=global_step, decay_steps=5000, decay_rate=0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients,var=zip(*optimizer.compute_gradients(loss))
    gradients_clipped, _ = tf.clip_by_global_norm(gradients, 1.25)
    opt=optimizer.apply_gradients(zip(gradients_clipped,var),global_step=global_step)
 
    # Validation
    val_output_after = tf.nn.softmax(RNN(val_input,val_output))
    val_probs = tf.nn.softmax(val_output_after)
 
    #add init op to the graph
    init = tf.initialize_all_variables()
    '''