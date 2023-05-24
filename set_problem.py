#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt
from function_generator import gp_generator

class set_problem(object):
    def __init__(self, num_ag, ker, l, x_range, x_count ,e_sigma):
        self.ker = ker
        self.num = num_ag
        self.e_sigma = e_sigma
        
        N = x_count # number of points to evaluate function
        self.x = np.linspace(x_range[0],x_range[1],N).reshape(-1,1)
        self.l = l # length scale of GP
        self.y_func = gp_generator(N, self.x, self.l, self.num, self.ker)
    
    def plot_function(self):
        plt.plot(self.x, self.y_func,'r')
        plt.show()
    
    def plot_sum_function(self):
        y_sum  = np.sum(self.y_func,axis =1) 
        plt.plot(self.x, y_sum,'r')
        plt.show()
       
    def sample_1D(self,i,x_in):
        pos = np.where(self.x==x_in)[0]
        return self.y_func[pos,i]
    
    def sum_sample_1D(self, x_in):
        pos = np.where(self.x==x_in)[0]
        y_sum  = np.sum(self.y_func,axis =1) 
        return y_sum[pos]
    
    def noisy_sample_1D(self, x_id):
        # pos = np.where(self.x==x_in)[0]
        e = np.random.normal(0.0, self.e_sigma, self.num)
        ag_pos = np.arange(0,self.num,1)
        return self.y_func[x_id,ag_pos] + e
               
    def best_reward_1D(self):  # Returning the highest function value
        y_sum  = np.sum(self.y_func,axis =1)        
        return np.max(y_sum)
    
    def best_action_1D(self):   #Returning the best action
        y_sum  = np.sum(self.y_func,axis =1)   
        return self.x[y_sum.argmax()]

        
        
        