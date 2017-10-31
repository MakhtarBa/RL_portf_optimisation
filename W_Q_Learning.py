# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:25:47 2017

@author: Makhtar Ba
"""

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

os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')

with open('Tickers.txt') as tickers:
    reader=tickers.read().split("\n")
    list_tickers=[read for read in reader]

data=pd.read_csv('AAPL.txt', sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]


data=data[" OPEN"]
data_df=pd.DataFrame(data)

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



dates=data_df.columns
lag=[(dates[i]-dates[i+1]).days for i in range (len(dates)-1)]
lag_dataframe = pd.DataFrame(lag)
lag_dataframe.plot(kind='hist')

plt.show()
return_df=pd.DataFrame()

for i in range(len(list(dates))-1):
    return_df[dates[i]]=data_df[dates[i+1]]/data_df[dates[i]]-1


##########################################


'''

Setting up the ICA 

'''

##########################################
return_df=return_df.transpose()

ica = FastICA(n_components=5)
Ind_components = ica.fit_transform(return_df)  # Reconstruct independ constituents 
A_ = ica.mixing_  # Get estimated mixing matrix

reconstitution_ICA=ica.inverse_transform(Ind_components) #recompose the returns from the ICA transformation
reconstitution_ICA=pd.DataFrame(reconstitution_ICA)

reconstitution_ICA.columns=return_df.columns

# We can `prove` that the ICA model applies by reverting the unmixing.
#assert np.allclose(data_df, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=5)
H = pca.fit_transform(return_df)  # Reconstruct signals based on orthogonal components
reconstitution_PCA= pca.inverse_transform(H)  # Reconstruct signals based on orthogonal components
reconstitution_PCA=pd.DataFrame(reconstitution_PCA)

reconstitution_PCA.columns=return_df.columns

    
# #############################################################################
# Plot results

plt.figure()

models = [return_df, reconstitution_ICA, reconstitution_PCA]

names = ['Observations (mixed signal)',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'black', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3,1, ii)
    plt.title(name)
    for sig, color in zip(model, colors):
        print(sig)
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()

# Comparison of the sumcum 

cumsum_ICA=pd.DataFrame(columns=reconstitution_ICA.columns)
cumsum_PCA=pd.DataFrame(columns=reconstitution_ICA.columns)
cumsum_return=pd.DataFrame(columns=reconstitution_ICA.columns)

for column in reconstitution_ICA.columns:
    cumsum_ICA[column]=reconstitution_ICA[column].cumsum()
    cumsum_PCA[column]=reconstitution_PCA[column].cumsum()
    cumsum_return[column]=return_df[column].cumsum()

plt.figure()
fig,ax=plt.subplots()
ax.plot(cumsum_ICA['AAPL'],color='orange')
ax.plot(cumsum_PCA['AAPL'],color='green')
ax.plot(cumsum_return['AAPL'],color='black')

plt.show()


##### Q-learning



'''

  Parameters 
 
'''
nu=0.01
num_bins=100
num_components=5
num_actions=2
gamma=0.9
epsilon=0.9
bins = np.array(np.arange(1,100,50))

# d is an index array holding the bin id for each point in A
d = np.digitize(return_df['MSFT'], bins)  
np.sum(d)

count, division =np.histogram(return_df['AAPL'], bins=100)
count_ICA, division_ICA =np.histogram(Ind_components[:,1], bins=100)

#Initialiwe the  value functions 

Q= np.array([np.random.randn(num_actions,num_components) for x in range(num_bins+1)])
        
def LocalizeBin(returns, component):
    global Ind_components
    count, division = np.histogram(Ind_components[:,component], bins=100)
    bin_num=0
    while returns>division[bin_num] and bin_num<100:
        bin_num+=1
    return bin_num

    
def Q_update(state,action, component,time):
    global Ind_components
    global Q
    global nu
    global gamma
    
    bin_num=LocalizeBin(state,component)
    count, division = np.histogram(Ind_components[:,component], bins=100)
    if bin_num <len(division)-1:    
        average_return= 0.5*(division[bin_num]+division[bin_num+1])
    else :
        average_return=division[bin_num]
    
    reward=log(1+action*Ind_components[time+1,component])
    Q[bin_num,action,component]=(1-nu)*Q[bin_num,action,component]+nu*(reward+gamma(max(Q[bin_num,:,component])))
   
    return Q
    


'''
def update_q(state, next_state, action, alpha, gamma):
    rsa = r[state, action]
    qsa = q[state, action]
    new_q = qsa + alpha * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    q[state][q[state] > 0] = rn
    return r[state, action]

'''

def show_traverse():
    # show all the transitions
    for i in range(len(q)):
        current_state = i
        traverse = "%i -> " % current_state
        n_steps = 0
        while current_state != 5 and n_steps < 20:
            next_state = np.argmax(q[current_state])
            current_state = next_state
            traverse += "%i -> " % current_state
            n_steps = n_steps + 1
        # cut off final arrow
        traverse = traverse[:-4]
        print("Greedy traversal for starting state %i" % i)
        print(traverse)
        print("")


#In the sates we have the time, the return and the fact of being short or long at the time 
        
# Core algorithm
'''
   Training_phase
'''

train=2000
for component in  range(len(Ind_components[0])):
    for t in range(train):
        return_component=Ind_components[t,component]
        state=LocalizeBin(return_component,component)
        
        if random.uniform(0,1)<epsilon:
            action=np.argmax(Q[state,:,component])

            #Should serve to visualiwe the transitions but for now is useless
            '''
            
            if iteration % int(num_iterations / 10.) == 0 and iteration>0:
                # Just to see the trajectory if we are doing enough exploration
                #pass
                show_traverse()
            '''    
            Q=Q_update(state,action, component,t)
        else:
            action=random.randint(0,1)
            Q=Q_update(state,action, component,t)
            

        
    
print(Q)
for t in range 
show_traverse()


