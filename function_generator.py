"""
Generating Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

def squared_normal(x,l):
    covar = np.exp(-(x-x.T)**2/(2*l**2))
    return covar

def rbf_torch(x,l):
    covar_mod = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    covar_mod.base_kernel.lengthscale = l
    covar_t = covar_mod(torch.from_numpy(x))
    return covar_t

def mattern_torch(x,l):
    nu = 2.5
    covar_mod = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu))
    covar_mod.base_kernel.lengthscale = l
    covar_t = covar_mod(torch.from_numpy(x))
    return covar_t

def gp_generator(N,xspace,l,num,ker):
    # Method 1
    # Using random multivariate normal distribution
    mean = np.zeros(N)
    # covar = squared_normal(x,l)
    if ker == "rbf":
        covar = rbf_torch(xspace,l).numpy()
    elif ker == "mat":
        covar = mattern_torch(xspace,l).numpy()
    y_point = np.random.multivariate_normal(mean, covar, num).T
    return y_point





