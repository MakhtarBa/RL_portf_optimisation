# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:21:57 2017

@author: Makhtar Ba
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
from math import *



def optimal_portfolio(returns,decision):
    global demixing
    '''
    :param decision: list of d_i_t telling if a long or a short position on component has been advised 
    :param returns:retunrs of the compoments up to time t 
    
    '''
    
    N_u=1
    N_l=1
    lower_bounds={component:0 for component in returns.columns}
    upper_bounds={component:0 for component in returns.columns}
    #print(upper_bounds)
    for component in returns.columns:    
        lower_bounds[component]=(1/2*(-1+tanh(decision[component]*N_l)))
        upper_bounds[component]=(1/2*(1+tanh(decision[component]*N_u)))
    
    
    # Convert to cvxopt matrices
    mu = opt.matrix(np.mean(returns, axis=0))
    demixing=opt.matrix(demixing)
    #Create Objective Matrix 
    #print(type(demixing),type(mu))
    P=(demixing)*mu
    S = opt.matrix(np.cov(returns.transpose()))
    risk_aversion=1
    #print(type(demixing),type(S))
    
    S=demixing*S*opt.matrix(np.transpose(demixing))
    '''
     Create constraint matrices
    '''
    
    #Weights boundaries 
    G_1 = opt.matrix(np.eye(np.shape(returns)[1]))   # negative n x n identity matrix
    G_2= -1*opt.matrix(np.eye(np.shape(returns)[1]))
    G=opt.sparse([[G_1,G_2]])
    
    h = opt.matrix([upper_bounds[i] for i in returns.columns]+[-1*lower_bounds[i] for i in returns.columns])
    
    #Sum of constraints equal to 1
    A = opt.matrix(1.0, (1, np.shape(returns)[1]))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    #print(type(S),type(P),type(G),type(h),type(A),type(b))
    portfolios = solvers.qp(S,P,G, h, A, b)
    return portfolios    

    
    

