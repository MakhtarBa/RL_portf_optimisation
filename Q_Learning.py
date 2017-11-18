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
from scipy.stats import bernoulli

import json 
        


##### Q-learning
        
def initLocalizeState(component):
    global granularity
    global Ind_components
    global k_it
    range_test=max(Ind_components[component])-min(Ind_components[component])
    number_state=int(range_test*granularity/np.std(Ind_components[component]))
    division=[np.std(Ind_components[component])*i/granularity for i in range(number_state+1)]
    return division

def LocalizeState(component_return,component,sub_division):
    global k_it
    bin_num=np.searchsorted(sub_division[component],component_return)
    bin_num=-1/2*(bin_num+len(sub_division[component]))*(k_it[component]-1)+bin_num*(k_it[component]+1)/2
    
    return int(bin_num)

    
def Q_update(state,action, component,time,nu):
    '''
       This function is used to implement the updates of the value functions 
    '''
    
    
    global Ind_components
    global Q
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
    index_component=list(Ind_components.columns).index(component)
    cost=log(1-delta*sum(demixing[index_component,:])*abs(k_it[component]-action))
    reward=log(1+action*Ind_components[component][time+1])+cost
    next_state=LocalizeState(reward,component,subdivision)
    Q[component][action,state]=(1-nu)*Q[component][action,state]+nu*(reward+gamma*(max(Q[component][:,next_state])))
    
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


#This definition conducts online policy learning it is more of a SARSA using an epsilon greedy technique but it should converge faster 

def Q_train(num_iterations,train,epsilon,sub_division):
    global Ind_components
    global Q
    global k_it
    global nu
    
    for iteration in range(num_iterations):  
        print ('{}'.format(iteration+1) +'out of {}'.format(num_iterations))
        for component in  Ind_components.columns: 
            for t in range(train):
                return_component=Ind_components[component][t]
                state=LocalizeState(return_component,component,sub_division)
                if random.uniform(0,1)<epsilon[iteration]:
                    
                    action=-1*(1-np.argmax(Q[component][:,state]))+np.argmax(Q[component][:,state])
        
                else:
                    rand=random.randint(0,1)
                    action=-1*(1-rand)+rand
                    
                Q=Q_update(state,action, component,t,nu[iteration,t])
                k_it[component]=action
    return Q



if __name__ == "__main__":
    os.chdir('C:/Users/Makhtar Ba/Documents/Columbia/TimeSeriesAnalysis/data/data')
    return_df=pd.read_csv('returns_df_BD.csv',index_col=0)
    Ind_components=pd.read_csv('Ind_components.csv',index_col=0)
    test_demixing=pd.read_csv('Demixing_matrix.csv')
    del test_demixing['Unnamed: 0']
    demixing=np.array(test_demixing)
    
    
    '''
    
    Testing the Fast ICA 
    
    '''
    
    
    corr_factors=np.corrcoef(Ind_components.transpose())
    corr_returns=np.corrcoef(return_df.transpose())
    cov_factors=np.cov(Ind_components.transpose())
    mean_factors=np.mean(Ind_components)
    #test_statistic= statsmodels.stats.diagnostic.acorr_ljungbox(Ind_components['0'])
    '''
       Test  choosing number of states
    '''
    '''
    number_state=[]
    for column in Ind_components.columns:
        range_test=max(Ind_components[column])-min(Ind_components[column])
        number_state.append(range_test*5/np.std(Ind_components[column]))
    plt.plot(number_state)
    '''
    
    
    
    '''
    
      Parameters 
     
    '''
    num_bins=100
    num_components=np.shape(return_df)[1]
    num_actions=2
    gamma=1-0.01/365
    bins = np.array(np.arange(1,100,50))
    num_iterations=50
    train_size=2000
    delta=0.005 # given in the paper 
    granularity=5
    epsilon=[0.25/(0.5*k+1) for k in range(num_iterations)]
    nu=np.array([[0.9*1/((1+0.01*k)*(i*0.05+1)) for k in  range(train_size)]for i in range(num_iterations)])
    
    #Model Initilization
    subdivision={component:0 for component in Ind_components.columns}
    
    for component in subdivision.keys():
            subdivision[component]=initLocalizeState(component)
    
    #k_it={component:-1*(1-init)+1*init for (init,component) in zip(list(bernoulli.rvs(0.5,size=Ind_components.shape[1])),Ind_components.columns)}
    k_it={component:-1*(1-int(init+0.5))+1*int(init+0.5) for (init,component) in zip([np.random.uniform(0,1) for size in Ind_components.columns],Ind_components.columns)}
    
    #Q= {component: np.random.randn(num_actions,2*len(subdivision[component])) for component in Ind_components.columns}
    Q= {component: 0.01*np.random.randn(num_actions,2*len(subdivision[component])) for component in Ind_components.columns}
    
    rewards={component:[] for component in Ind_components.columns}
    portf_return=[]

    Q=Q_train(num_iterations,train_size,epsilon,subdivision)
    asset_portf_return=[]
    factor_portf_return=[]
    asset_equally_weighted=[]
    factor_equally_weighted=[]
    asset_weights=[]
    factor_weights=[]
                
    lower_bounds={component:[] for component in Ind_components.columns}
    upper_bounds={component:[] for component in Ind_components.columns}

    for t in range(train_size,len(return_df)):
        print('{}'.format(t) +' out of {}'.format(len(return_df)))
        
        decision={component:0 for component in Ind_components.columns}
        for component in Ind_components.columns: 
            return_component=Ind_components[component][t-1]
            state=LocalizeState(return_component,component,subdivision)
            action=-1*(1-np.argmax(Q[component][:,state]))+np.argmax(Q[component][:,state])
            
            decision[component]=(Q[component][1,state]-Q[component][0,state])
            lower_bounds[component].append((1/2*(-1+tanh(decision[component]*N_l))))
            upper_bounds[component].append((1/2*(1+tanh(decision[component]*N_u))))        
        
            rewards[component].append(log(1+((action+1)/2-(1-action)*1/2)*Ind_components[component][t]))
            
        portfolio=optimal_portfolio(Ind_components.ix[:t,:],decision,demixing)      
        weights=portfolio['x']
        print(sum(weights))
    
        
            
        asset_weights.append(np.dot(weights.T,demixing))
        factor_weights.append(list(weights))
        #print(np.shape(weights))
        factor_actual_returns=opt.matrix(Ind_components.ix[t,:])
        asset_actual_returns=opt.matrix(return_df.ix[t,:])
        print(np.size(asset_weights),t-train_size)
        asset_portf_return.append(np.dot(asset_actual_returns.T,(asset_weights[t-train_size]).T)[0][0])
        factor_portf_return.append(np.dot(factor_actual_returns.T,np.array(weights))[0][0])
        asset_equally_weighted.append(1/(np.shape(Ind_components)[1])*sum(asset_actual_returns))
        factor_equally_weighted.append(1/(np.shape(Ind_components)[1])*sum(factor_actual_returns))
        k_it={component:np.sign(x) for (component,x) in zip(Ind_components.columns,weights)}
        
    
N_u=100
N_l=100
for i in range()   
plt.plot([factor_weights[i][10] for i in range(len(factor_weights))])
plt.plot([upper_bounds['10'][i] for i in range(len(upper_bounds['10']))])
plt.plot([lower_bounds['10'][i] for i in range(len(lower_bounds['10']))])

sum([0 if (upper_bounds['10'][i]>factor_weights[i][10]) else 1 for i in range(len(upper_bounds['10']))  ])
sum([0 if (lower_bounds['10'][i]<factor_weights[i][10]) else 1 for i in range(len(upper_bounds['10']))  ])
        
cumsum_eq_weight=pd.DataFrame(asset_equally_weighted).cumsum()
cumsum_portf=pd.DataFrame(asset_portf_return).cumsum()
pd.DataFrame(asset_portf_return).hist()
plt.plot(asset_equally_weighted)
plt.plot(asset_portf_return)
plt.show()
plt.plot(factor_weights)

# Cumsums plots
plt.plot(cumsum_eq_weight)
plt.plot(cumsum_portf)
plt.show()

asset_weights_df=pd.DataFrame(asset_weights)


