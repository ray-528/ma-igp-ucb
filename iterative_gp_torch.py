#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental GP
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gpytorch
from time import time
from tqdm import tqdm
import gc

torch.set_default_dtype(torch.float64)
nu =2.5

class GP_model(nn.Module):
    def __init__(self, nagent, ker, e_sig, xspace, epochs, l,cuda_a):
        
        super().__init__()
        self.ker = ker
        self.nagent = nagent
        self.cuda_a = cuda_a
        if cuda_a=='yes':
            self.lamda = torch.tensor([e_sig**2])[:, None, None].cuda()
            self.space = torch.from_numpy(xspace).cuda() # state space x
        
            size = xspace.shape[1]
            self.size  = size # size of discretised state x
            self.T = epochs # Total number of sample

            self.k_T = torch.zeros([nagent, size, epochs]).cuda()
            self.K_T =  torch.zeros([nagent, epochs, epochs]).cuda()
            self.l=l

            # define kernel matrix function
            if ker=="rbf":
                self._kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([nagent])),
                    batch_shape=torch.Size([nagent])).cuda()
            elif ker=="mat":
                self._kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu,batch_shape=torch.Size([nagent])),
                    batch_shape=torch.Size([nagent])).cuda()
                
            self._kernel.base_kernel.lengthscale = l
            self.kernel = lambda x1, x2: self._kernel(x1, x2).evaluate()

            self.k = self.kernel(self.space, self.space) #[0][None,:,:]
            self.K_inv = torch.zeros([nagent, epochs, epochs]).cuda()
        else: 
            self.lamda = torch.tensor([e_sig**2])[:, None, None]
            self.space = torch.from_numpy(xspace) # state space x

            size = xspace.shape[1]
            self.size  = size # size of discretised state x
            self.T = epochs # Total number of sample

            self.k_T = torch.zeros([nagent, size, epochs])
            self.K_T =  torch.zeros([nagent, epochs, epochs])
            self.l=l

            # define kernel matrix function
            if ker=="rbf":
                self._kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([nagent])),
                    batch_shape=torch.Size([nagent]))
            elif ker=="mat":
                self._kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu,batch_shape=torch.Size([nagent])),
                    batch_shape=torch.Size([nagent]))
                
            self._kernel.base_kernel.lengthscale = l
            self.kernel = lambda x1, x2: self._kernel(x1, x2).evaluate()

            self.k = self.kernel(self.space, self.space) #[0][None,:,:]
            self.K_inv = torch.zeros([nagent, epochs, epochs])

    def forward(self, x_train, y_train, t):
        # t is the total number of current samples
        self.k_T[:,:,t] = self.kernel(self.space, x_train[:,t:])[:,:,0]
        self.get_ker_inv(x_train,t)
        
        temp = torch.bmm(self.k_T[:,:,:t+1], self.K_inv[:,:t+1,:t+1])
        mean = torch.bmm(temp, y_train)[:,:,0]
        variance = self.k - torch.bmm(temp, self.k_T[:,:,:t+1].transpose(1,2))
        sigma = torch.diagonal(variance,dim1=1, dim2=2).sqrt()

        del temp
        del variance
        if self.cuda_a=='yes':
            return mean.cpu().numpy(), sigma.cpu().numpy()
        else:
            return mean.numpy(), sigma.numpy()

    def get_ker_inv(self, x_train, t):

        if t==0:
            self.K_T[:,0,0] = self.kernel(x_train, x_train)[:,0,0] + self.lamda
            self.K_inv[:,0,0] = (1/self.K_T[:,0,0])

            return
    
        K_inv_old = self.K_inv[:,:t,:t].clone() # shape = (nagent, t-1, t-1)
        b = self.kernel(x_train[:,:-1], x_train[:,-1:]) # shape = (na, t-1, 1)
        d = self.kernel(x_train[:, -1:], x_train[:,-1:]) + self.lamda # shape = (na, 1, 1)
        e = gpytorch.matmul(K_inv_old, b) # shape = (na, t-1, 1)
        g = 1/(d - gpytorch.matmul(b.transpose(1,2), e)) # shape = (na, 1, 1)
        
        self.K_inv[:,:t,:t] =  K_inv_old + g*torch.bmm(e, e.transpose(1,2)) 
        self.K_inv[:,t,:t] = (-g*e)[:,:,0]
        self.K_inv[:,:t,t] = (-g*e)[:,:,0]
        self.K_inv[:,t,t] = g[:,0,0]

        del K_inv_old
        del b
        del d
        del e
        del g

