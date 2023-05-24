# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm
from iterative_gp_torch import GP_model
import matplotlib.pyplot as plt

class MAB(object):

  def __init__(self, problem, Adj, K, epoch, ker,cuda_a, l, e_sigma, consensus, C):
    self.meshgrid = problem.x
    self.problem = problem
    self.num = K # number of agents
    self.Adj = Adj # Adjacency matrix
    self.beta = 100 # Just initializing the beta
    self.ker = ker
    self.cuda_a = cuda_a
    self.epoch = epoch
    
    self.X_grid = self.meshgrid
    
    with torch.no_grad():
        xspace_gp = self.X_grid.reshape(1, -1, 1) .repeat(K, axis=0)
        self.gpmodel = GP_model(K, ker, e_sigma, xspace_gp, epoch, l, cuda_a)

    # mean and variance of own function estimated!
    self.mu = np.zeros([K,len(self.meshgrid)])
    self.sigma = 1*np.ones([K,len(self.meshgrid)])
    # mean and variance at previous time step!
    self.mu_p = self.mu
    self.sigma_p = self.sigma
    # estimated mean and variance of complete function!
    self.mu_h = self.mu
    self.sigma_h = self.sigma    
    self.action = np.zeros([K,epoch])-10
    self.reward =  np.zeros([K,epoch])-10

    self.consensus = consensus
    if consensus == "central":
        self.consensus_alg = self.centralized
    else:
        self.consensus_alg =  self.running_private
    
    self.mu_ha = self.mu
    self.sigma_ha = self.sigma
    self.mu_hb = self.mu
    self.sigma_hb = self.sigma
    self.C = C # number of iterations in a stage 

  def update_beta(self,t):
    '''
    Updating the value of beta at each iteration
    '''
    delta = 0.1; B = 5; R = 0.2
    gamma = 0.2*(np.log(t+1))**2
    self.beta = B + R*np.sqrt(( gamma + 1 + np.log(1/delta)))

  def learn(self):
    for t in tqdm(range(self.epoch)):
        # Deciding best beta
        self.update_beta(t)
        self.sample(t) # All agents making a selection and sampling

        with torch.no_grad():
            x_train = self.action[:,:t+1].reshape(self.num,-1,1)
            
            y_train = self.reward[:,:t+1].reshape(self.num,-1,1)
            if self.cuda_a=='yes':
                x_train = torch.from_numpy(x_train).cuda()
                y_train = torch.from_numpy(y_train).cuda()
            else: 
                x_train = torch.from_numpy(x_train)
                y_train = torch.from_numpy(y_train)
            mean, sigma = self.gpmodel(x_train, y_train, t)
            del x_train
            del y_train
        self.mu_p, self.sigma_p = self.mu, self.sigma
        self.mu, self.sigma = mean, sigma
        self.consensus_alg()

  def sample(self,t):
    '''
    Sampling the local functions
    t = current time step
    '''
    sam_id = np.argmax(self.mu_h + self.sigma_h*self.beta,axis=1)
    sam_pt = self.X_grid[sam_id]
    rew = self.problem.noisy_sample_1D(sam_id)
    self.action[:,t] = sam_pt.flatten()
    self.reward[:,t] = rew.flatten()

  def learn_delayed(self):
    stage=0
    t = 0
    num_stage = int(self.epoch/self.C)
    for stage in tqdm(range(num_stage)):
        if stage >=2:
            self.mu_ha = self.mu_hb 
            self.sigma_ha = self.sigma_hb 
        if stage >=1:
            self.mu_hb = self.mu_h
            self.sigma_hb =  self.sigma_h 
        count = 0
        while count < self.C:
            # Deciding best beta
            self.update_beta(t)
            self.sample_delayed(t,stage) # All agents making a selection and sampling
            with torch.no_grad():
                x_train = self.action[:,:t+1].reshape(self.num,-1,1)
                y_train = self.reward[:,:t+1].reshape(self.num,-1,1)
                x_train = torch.from_numpy(x_train).cuda()
                y_train = torch.from_numpy(y_train).cuda()
                mean, sigma = self.gpmodel(x_train, y_train, t)
                del x_train
                del y_train
                
            self.mu_p = self.mu
            self.sigma_p = self.sigma
            self.mu = mean
            self.sigma = sigma
            self.consensus_alg()

            self.mix_b()  
            count = count+1
            t = t+1

  def sample_delayed(self,t,stage):
    if stage<2:
        sam_id = np.argmax(self.mu_h + self.sigma_h*np.sqrt(self.beta),axis=1)
    else:
        sam_id = np.argmax(self.mu_ha + self.sigma_ha*np.sqrt(self.beta),axis=1)
    sam_pt = self.X_grid[sam_id]
    rew = self.problem.noisy_sample_1D(sam_id)
    self.action[:,t] = sam_pt.flatten()
    self.reward[:,t] = rew.flatten()

##______Consensus algorithms__________________________________
    
  def centralized(self):
    avg_mu = np.sum(self.mu_h,axis=0).reshape(1,-1).repeat(self.num,axis=0)  
    self.mu_h = avg_mu/self.num + self.mu - self.mu_p
    avg_sig = np.sum(self.sigma_h,axis=0).reshape(1,-1).repeat(self.num,axis=0)
    self.sigma_h = avg_sig/self.num + self.sigma - self.sigma_p

  def running_private(self):
     self.mu_h = self.Adj@ self.mu_h + self.mu - self.mu_p
     self.sigma_h= self.Adj@ self.sigma_h + self.sigma - self.sigma_p

  def mix_b(self): #mixing beta
     self.mu_hb = self.Adj@ self.mu_hb
     self.sigma_hb = self.Adj@ self.sigma_hb
         
  def independent(self):
    self.mu_h = self.mu
    self.sigma_h = self.sigma
    
