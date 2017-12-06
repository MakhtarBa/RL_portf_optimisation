
# coding: utf-8


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


return_df=pd.DataFrame()

for i in range(len(list(dates))-1):
    return_df[dates[i]]=data_df[dates[i+1]]/data_df[dates[i]]-1

return_df.transpose()

look_back=3
hidden_dim = 2

a=np.array([return_df['AAPL'][j:j+look_back].values for j in range(len(return_df['AAPL'])-look_back)],np.float32)
#a=tf.cast(a, tf.float32)

'''
t1=np.concatenate((a[:3],[1]))
t2=np.concatenate((a[3:6],[1]))


t1=np.ndarray.tolist(t1)
t2=np.ndarray.tolist(t2)
a=np.array([t1,t2])

'''


time_horizon=1000

batch_size=100

reg_tf = tf.constant(0.01)

x1_tf = tf.placeholder(tf.float32,name='one_variable', shape=(None,look_back+1))
#y_tf = tf.placeholder(tf.int64, shape=(None,))

W1_tf = tf.Variable(1e-2*np.random.rand(look_back+1, hidden_dim).astype('float32'))
W2_tf=W1_tf
h1_tf = tf.nn.tanh(tf.matmul(x1_tf, W1_tf) )
softmax = tf.nn.softmax(logits= h1_tf )
#L1_loss = tf.nn.l2_loss(W1_tf)
loss1_tf = tf.reduce_mean(softmax) 


x2_tf= tf.placeholder(tf.float32, name='two_variable',shape=(None,look_back+1))
h2_tf = tf.nn.tanh(tf.matmul(x2_tf, W1_tf) )
softmax2 = tf.nn.softmax(logits= h2_tf )

loss2_tf = tf.reduce_mean(softmax2)



x3_tf= tf.placeholder(tf.float32,name='three_variable', shape=(None,look_back+1))

h3_tf = tf.nn.tanh(tf.matmul(x3_tf, W1_tf) )
softmax3 = tf.nn.softmax(logits= h3_tf )
L3_loss = tf.nn.l2_loss(W1_tf)
loss3_tf = tf.reduce_mean(softmax3)+reg_tf*L3_loss

x3_tf= tf.placeholder(tf.float32,name='three_variable', shape=(None,look_back+1))

h3_tf = tf.nn.tanh(tf.matmul(x3_tf, W1_tf) )
softmax3 = tf.nn.softmax(logits= h3_tf )
L3_loss = tf.nn.l2_loss(W1_tf)
loss3_tf = tf.reduce_mean(softmax3)+reg_tf*L3_loss


learning_rate = 0.1

optimizer=tf.train.GradientDescentOptimizer(learning_rate)
grad_operations=optimizer.compute_gradients(loss3_tf,var_list=W1_tf)
grad_calcul=optimizer.apply_gradients(grad_operations)
grad=tf.gradients(loss3_tf,W1_tf)

    

init = tf.global_variables_initializer()
np.set_printoptions(precision=3)

with tf.Session() as sess:
    
    sess.run(init)
    print('W1_tf is ')
    sess.run(W1_tf)
    print('W2_tf is ')
    sess.run(W2_tf)
    #output = sess.run(train, feed_dict={x_tf: a})
    
    input_1=np.concatenate((a[0],[1])).reshape(1,4)
    input_2=np.concatenate((a[1],[1])).reshape(1,4)
    input_3=np.concatenate((a[2],[1])).reshape(1,4)
    
    '''
    input_1=tf.reshape(tf.concat((a[0],[1]),axis=0),[1,4])
    input_2=tf.reshape(tf.concat((a[1],[1]),axis=0),[1,4])
    input_3=tf.reshape(tf.concat((a[2],[1]),axis=0),[1,4])
    '''
    grad_vals=sess.run(grad,feed_dict={x1_tf:input_1 ,x2_tf:input_2,x3_tf:input_3})
    grad_=sess.run(grad_calcul, feed_dict={x1_tf:input_1 ,x2_tf:input_2,x3_tf:input_3})     
    print(grad_vals)
    print(grad_)
    update=sess.run(tf.norm(grad_vals[0]*learning_rate))
    test=sess.run(tf.norm(W1_tf)*0.0000000001)
    
    count=0
    loss_value=[]
    while(count<10):
        count+=1
        grad=tf.gradients(loss3_tf,W1_tf)
        '''
        input_1=tf.reshape(tf.concat((a[0],[1]),axis=0),[1,4])
        input_2=tf.reshape(tf.concat((a[1],[1]),axis=0),[1,4])
        input_3=tf.reshape(tf.concat((a[2],[1]),axis=0),[1,4])
        '''
        
        input_1=np.concatenate((a[0],[1])).reshape(1,4)
        input_2=np.concatenate((a[1],[1])).reshape(1,4)
        input_3=np.concatenate((a[2],[1])).reshape(1,4)
        
        grad_operations_val , loss_tf_val, grad_calcul_val = sess.run([grad_operations, loss3_tf, grad_calcul], feed_dict={x1_tf:input_1 ,x2_tf:input_2,x3_tf:input_3})
        grad_val=sess.run(grad, feed_dict={x1_tf:input_1 ,x2_tf:input_2,x3_tf:input_3})
        
        update=sess.run(tf.norm(grad_val[0]*learning_rate))
        W2_tf=tf.add(W2_tf,grad_val[0]*0)
        loss_value.append(loss_tf_val)
        print(count)
        



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[98]:

output


# In[27]:

hidden_dim = 2
reg_tf = tf.constant(0.01)

x_tf = tf.placeholder(tf.float32, shape=(None, 3072))
y_tf = tf.placeholder(tf.int64, shape=(None,))

W1_tf = tf.Variable(1e-2*np.random.rand(3072, hidden_dim).astype('float32'))
h1_tf = tf.nn.log_sigm(tf.matmul(x_tf, W1_tf) + b1_tf)

 = tf.nn.softmax_cross_entropy_with_logits(logits= h2_tf, labels=tf.one_hot(y_tf,10))
L2_loss = tf.nn.l2_loss(W1_tf) + tf.nn.l2_loss(W2_tf)
loss_tf = tf.reduce_mean(cross_entropy) + reg_tf*L2_loss 

init = tf.global_variables_initializer()
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_tf)
correct_prediction = tf.equal(y_tf, tf.argmax(h2_tf,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_train = 49000
batch_size = 500
num_batch = num_train//batch_size
with tf.Session() as sess:
    sess.run(init)
    for e in range(10):
        for i in range(num_batch):
            batch_xs, batch_ys = X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            sess.run(train_step, feed_dict={x_tf: batch_xs, y_tf: batch_ys})
        val_acc = sess.run(accuracy, feed_dict={x_tf: X_val, y_tf: y_val})
        print('epoch {}: valid acc = {}'.format(e+1, val_acc))
    
    test_acc = sess.run(accuracy, feed_dict={x_tf: X_val, y_tf: y_val})
    print('test acc = {}'.format(test_acc))


# In[31]:

tf.gradients(softmax,W1_tf)


# In[ ]:



