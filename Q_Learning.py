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
import random
from math import *
os.chdir('C:/Users/Makhtar Ba/Documents/GitHub/RL_portf_optimisation')
    
from optimization import *

import statsmodels



        


##### Q-learning
        
def LocalizeState(returns, component):
    global Ind_components
    global k_it
    count, division = np.histogram(Ind_components[component], bins=100)
    bin_num=0
    while returns>division[bin_num] and bin_num<100 :
        bin_num+=1
    bin_num=-1/2*(bin_num+100)*(k_it[component]-1)+bin_num*(k_it[component]+1)/2
    return bin_num

    
def Q_update(state,action, component,time):
    '''
       This function is used to implement the updates of the value functions 
    '''
    
    
    global Ind_components
    global Q
    global nu
    global gamma
    global k_it
    global delta
    global demixing
    
    '''
    The next part is commented because its not used anymore but can be useful in the future 
    as a first aproach to the value function with consists of using the mean return as the 
    approximation of the next return 
    '''
    
    '''
    if bin_num <len(division)-1:    
        average_return= 0.5*(division[state]+division[state+1])
    else :
        average_return=division[state]
    '''
    #print(sum(demixing[component,:]),k_it[component]-action)
    cost=log(1-delta*sum(demixing[component,:])*abs(k_it[component]-action))
    reward=log(1+action*Ind_components[component][time+1])+cost
    next_state=LocalizeState(reward,component)
    
    Q[state,action,component]=(1-nu)*Q[state,action,component]+nu*(reward+gamma*(max(Q[next_state,:,component])))
    
    return Q
    



def show_traverse(component):
    '''
       This function highlights the different transitions in the state space the idea
       was to use it to make sure that we have enough exploration
    '''
    
    global Q
    
    # show all the transitions
    for i in range(len(Q)):
        current_state = i
        traverse = "%i -> " % current_state
        n_steps = 0
        while current_state != 5 and n_steps < 20:
            next_action = np.argmax(Q[state,:,component])
            next_state=state+1
            current_state = next_state
            traverse += "%i -> " % current_state
            n_steps = n_steps + 1
        # cut off final arrow
        traverse = traverse[:-4]
        print("Greedy traversal for starting state %i" % i)
        print(traverse)
        print("")



def Q_train(num_iterations,train,epsilon):
    global Ind_components
    global Q
    global k_it
    
    for iteration in range(num_iterations):  
        print ('{}'.format(iteration+1) +'out of {}'.format(num_iterations))
        for component in  Ind_components.columns:
            for t in range(train):
                return_component=Ind_components[component][t]
                state=LocalizeState(return_component,component)
                
                if random.uniform(0,1)<epsilon:
                    
                    action=-1*(1-np.argmax(Q[state,:,component]))+np.argmax(Q[state,:,component])
        
                    #Should serve to visualiwe the transitions but for now is useless
                    '''
                    
                    if iteration % int(num_iterations / 10.) == 0 and iteration>0:
                        # Just to see the trajectory if we are doing enough exploration
                        #pass
                        show_traverse()
                    '''
                else:
                    rand=random.randint(0,1)
                    action=-1*(1-rand)+rand
                    
                Q=Q_update(state,action, component,t)
                k_it[component]=action
    return Q



if __name__ == "__main__":
    os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')
    return_df=pd.read_csv('returns_df_BD.csv',index_col=0)
    Ind_components=pd.read_csv('Ind_components.csv',index_col=0)
    
    '''
    
    Setting up the ICA 
    
    '''
    
    ##########################################
    
    '''
    
    Testing the Fast ICA 
    
    '''
    
    ica = FastICA(n_components=5)
    Test_Ind_components=ica.fit_transform(return_df)  # Reconstruct independ constituents 
    Test_Ind_components=pd.DataFrame(Test_Ind_components)
    mixing = ica.mixing_  # Get estimated mixing matrix
    demixing=np.linalg.inv(mixing)
    
    corr_factors=np.corrcoef(Ind_components.transpose())
    corr_returns=np.corrcoef(return_df.transpose())
    cov_factors=np.cov(Ind_components.transpose())
    mean_factors=np.mean(Ind_components)
    test_statistic= statsmodels.stats.diagnostic.acorr_ljungbox(Ind_components['0'])
    
    
    plt.plot(return_df['AAPL'])
    Test_Ind_components.describe()
    test_corr_factors=np.corrcoef(Test_Ind_components.transpose())
    test_corr_returns=np.corrcoef(return_df.transpose())
    test_cov_factors=np.cov(Test_Ind_components.transpose())
    test_mean_factors=np.mean(Test_Ind_components)
    
    plt.show()       
    
    normal_test_1=[np.random.multivariate_normal([1,1,1],np.eye(3)) for s in range(3)]
    ica = FastICA(n_components=3)
    Test_Ind_components=ica.fit_transform(normal_test_1)  # Reconstruct independ constituents 
    Test_Ind_components=pd.DataFrame(Test_Ind_components)
    mixing = ica.mixing_  # Get estimated mixing matrix
    demixing=np.linalg.inv(mixing)
    
    
    '''
    
      Parameters 
     
    '''
    nu=0.01
    num_bins=100
    num_components=np.shape(return_df)[1]
    num_actions=2
    gamma=0.09
    epsilon=0.09
    bins = np.array(np.arange(1,100,50))
    num_iterations=100
    train_size=2000
    delta=0.05
    #Model Initilization
    
    k_it={component:-1*(1-init)+1*init for (init,component) in zip([random.uniform(0,1) for component in Ind_components.columns],Ind_components.columns)}
    Q= np.array([0.1*np.random.randn(num_actions,num_components) for x in range(2*(num_bins)+1)])
    
    rewards={component:[] for component in Ind_components.columns}
    portf_return=[]
    equally_weighted=[]
    for t in range(train_size,len(return_df)):
        print('{}'.format(t) +' out of {}'.format(len(return_df)))
        decision={component:0 for component in Ind_components.columns}
        for component in Ind_components.columns:
            return_component=Ind_components[component][t-1]
            state=LocalizeState(return_component,component)
            action=-1*(1-np.argmax(Q[state,:,component]))+np.argmax(Q[state,:,component])
            decision[component]=(Q[state,1,component]-Q[state,0,component])
            rewards[component].append(log(1+(action-(1-action)*1)*Ind_components[component][t]))
        portfolio=optimal_portfolio(Ind_components.ix[:t,:],decision)      
        weights=portfolio['x']
        #print(np.shape(weights))
        actual_returns=opt.matrix(Ind_components.ix[t,:])
        portf_return.append(blas.dot(actual_returns,weights))
        equally_weighted.append(1/(np.shape(Ind_components)[1])*sum(actual_returns))
        k_it={component:np.sign(x) for (component,x) in zip(Ind_components.columns,weights)}
        
        

cumsum_eq_weight=pd.DataFrame(equally_weighted).cumsum()
cumsum_portf=pd.DataFrame(portf_return).cumsum()

plt.plot(equally_weighted)
plt.plot(portf_return)
plt.show()

# Cumsums plots
plt.plot(cumsum_eq_weight)
plt.plot(portf_return)
plt.show()

