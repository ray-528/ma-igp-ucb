#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt

def plot2D_agent(alg, consensus,
                save_fig, fol_name, graph,ker):
    plt.figure()
    x = alg.meshgrid
    estimate = alg.mu
    prob = alg.problem
    for i in range(alg.num):
        plt.figure()
        plt.plot(x, estimate[i], alpha=0.5, color='r', label= "Estimated")
        plt.plot(x, prob.y_func[:,i], alpha=0.5, color='k', label = "Actual")
        plt.scatter(alg.action[i], alg.reward[i], c='g', marker='o', alpha=1.0,label ="Sampled points")
        plt.legend()
        # plt.title('Individual learning of agents'+str(i+1))
        plt.ylabel('$f_'+str(i+1)+'(x)$')
        plt.xlabel('$x$')
        if save_fig =="yes":
            plt.savefig(f"{fol_name}/{consensus}_{graph}-graph_{ker}_{alg.num}ag_local{i}.png", dpi = 600)
        else:
            plt.show()

def plot2D_network(alg, consensus, save_fig, fol_name, graph,ker):
    x = alg.meshgrid
    prob = alg.problem
    y_sum  = np.sum(prob.y_func,axis =1) 
    col = ['b', 'c', 'g', 'm','r',]    
    plt.figure()
    plt.plot(x, y_sum/alg.num, alpha=1, color='k', label = "Actual")
    # plt.plot(x, alg.mu_h[0], alpha=0.5, color=col[0])
    for i in range(alg.num): 
        plt.plot(x, alg.mu_h[i], col[i], alpha=1,  label = "Estimate"+str(i+1))
        plt.scatter(alg.action[i], alg.reward[i], marker='o', alpha=1.0)
    # plt.savefig(f'Plot_{self.Model}_{consensus}_agent{i}_network_estimated_learning.pdf')
    plt.legend()
    # plt.title('Network learning')
    plt.ylabel('$ 1/N \sum f(x)$')
    plt.xlabel('$x$')
    if save_fig =="yes":
        plt.savefig(f"{fol_name}/{consensus}_{graph}-graph_{ker}_{alg.num}ag_sampled-points.png", dpi = 600)
    else:
        plt.show()



