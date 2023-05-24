# -*- coding: utf-8 -*-
"""
Distributed Optimization via Kernelized Multi-armed Bandits
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat

from set_network import set_net
from set_problem import set_problem
from MA_GPUCB import MAB
from plot_analysis import plot2D_agent, plot2D_network

from plot_main import plot_group_reward
from plot_main import plot_cumu_regret
import random
import torch

def bandit_ma_gp(consensus,num_epoch, run, graph,ker, save_fig, 
                 fol_name,cuda_a, x_range, x_count,l, e_sigma, C):
    '''
    Runs the distributed optimization algorithm

    :param consensus: Algorithm to use
    :param num_epoch: Number of iterations
    :param run: Run number
    :param graph: Network to use
    :param ker: Kernel to use
    :param save_fig: Should the data be saved
    :param fol_name: Folder name where data should be saved
    :param cuda_a: Is cuda available?
    :param x_range: Range of action space
    :param x_count: Number of elements in action space
    :param l: kernel hyper-parameter
    :param e_sigma: Measurement noise
    :param C: Hyper-parameter for MAD-IGP-UCB
    '''
    
    net = set_net() 
    # get adjacency matrix
    if graph =='random':
        num_ag = 100
        Adj = net.big_random(num_ag)
    elif graph =='small':
        num_ag = 5
        Adj = net.small()
    elif graph =='facebook':
        Adj = net.big_fb(run)
        num_ag = 100
    elif graph =='twitch':
        Adj = net.big_twitch(run)
        num_ag = 100
    elif graph =='skitter':
        Adj = net.big_skitter(run)
        num_ag = 100
    
    # Collecting data over multiple runs
    ins_reg_over_runs = np.empty([num_epoch,0]) # Instantaneous regret r_t
    avg_rew_over_runs = np.empty([num_epoch,0]) # Average reward
    cumu_regret_over_runs = np.empty([num_epoch,0]) # Cumulative regret R_t
    cumu_regret_pertime_runs = np.empty([num_epoch,0]) # Cumulative regret per time R_t/t
    
    '''
    Learning MA-GP-UCB algorithm
    '''
    print("Algo:",consensus," Num agent:",num_ag," kernel:",ker, " run:", run)
    problem = set_problem(num_ag, ker, l, x_range, x_count, e_sigma)
    # Learning and sampling for n epochs
    # problem.plot_function()
    # problem.plot_sum_function()
    best = problem.best_reward_1D()
    func_max = best
    print("Best reward:", best)
    
    solution = MAB(problem, Adj, num_ag, num_epoch, ker,cuda_a,l,e_sigma, consensus, C)
    if consensus == "delayed":
        solution.learn_delayed()
    else:
        solution.learn()
        
    # Calculating Network rewards_____________________________________
    avg_reward = np.zeros(num_epoch)
    for i in range(num_epoch):
        temp = 0
        for j in range(num_ag):
            temp = temp + problem.sum_sample_1D(solution.action[j][i])
        avg_reward[i] = temp/num_ag # F(x1)+F(x2)+... /n

    # Plotting points sampled______________________________________________
    plot2D_agent(solution, consensus, save_fig, fol_name, graph,ker)
    plot2D_network(solution, consensus, save_fig, fol_name, graph,ker)

    # Instantaneous regret_________________________________________________
    ins_regret = best - avg_reward
    
    # Cumulative regret__________________________________________________
    cumu_regret = np.zeros(num_epoch)
    cumu_regret_pertime = np.zeros(num_epoch)
    cumu_regret[0] = ins_regret[0]
    cumu_regret_pertime[0] = ins_regret[0]
    for i in range(1,num_epoch):
        cumu_regret[i] = cumu_regret[i-1] + ins_regret[i] 
        cumu_regret_pertime[i] = cumu_regret[i]/(i+1)
    
    # Collecting data over multiple runs
    ins_reg_over_runs = np.hstack((ins_reg_over_runs, ins_regret.reshape([-1,1])))
    avg_rew_over_runs = np.hstack((avg_rew_over_runs, avg_reward.reshape([-1,1])))
    cumu_regret_over_runs = np.hstack((cumu_regret_over_runs,cumu_regret.reshape([-1,1])))
    cumu_regret_pertime_runs = np.hstack((cumu_regret_pertime_runs,cumu_regret_pertime.reshape([-1,1])))
    torch.cuda.empty_cache()
    
    # plt.figure()    
    plot_group_reward(consensus,num_epoch,avg_rew_over_runs,
                      save_fig, fol_name,graph,ker, num_ag)
    plot_cumu_regret(consensus,num_epoch,cumu_regret_over_runs,
                     save_fig, fol_name, graph,ker, num_ag)

    if save_fig == 'yes':
        plt.close('all')
        
    time = np.linspace(1,num_epoch,num_epoch)
    
    mdic = {"consensus": consensus, "num_ag": num_ag,
            "graph":graph, "ker":ker,
            "ins_reg_over_runs": ins_reg_over_runs,
            "avg_rew_over_runs": avg_rew_over_runs,
            "cumu_regret_over_runs": cumu_regret_over_runs,
            "cumu_regret_pertime_runs":cumu_regret_pertime_runs,
            "time" : time, "func_max": func_max,
            "num_epoch": num_epoch, "num_runs" : run, "C" : C}
    
    if save_fig == 'yes':
        savemat(f"{fol_name}/{consensus}_{graph}-graph_{ker}_{num_ag}ag_{num_epoch}epoch_run{run}.mat", mdic)
    

if __name__ == "__main__":
    '''
    Consensus type (Algorithms)-
    - central
    - private
    - delayed
    
    graph (Networks)-
    - small (5 agent)
    - random (Large- Erdos-Renyi)
    - facebook (Large)
    - twitch (Large)
    - skitter (Large)
    
    Kernels-
    - rbf (Sqaured exponential)
    - mat (Mattern kernel)
    '''
    #Hyper-parameters
    x_range = [0,1]
    x_count = 100
    e_sigma = 0.2
    l = 0.1
    C = 2 # number of iterations in a stage

    save_fig = 'yes'
    fol_name = "plots"
    num_epoch = 1000 # Number of runs in single episode
    run =  2 # Experiments number to study the mean behaviour

    graph = "small"
    cuda_a = "yes"
    seed = 500 +run
    np.random.seed(seed)
    random.seed(seed)
    ker = 'rbf' #kernel

    # consensus = "central"
    # bandit_ma_gp(consensus,num_epoch, run, graph,ker, 
    #             save_fig, fol_name,cuda_a, x_range, x_count, l, e_sigma, C)
    
    consensus = "private"
    bandit_ma_gp(consensus,num_epoch, run, graph,ker, 
                 save_fig, fol_name,cuda_a, x_range, x_count, l, e_sigma, C)
    
    # consensus = "delayed"
    # bandit_ma_gp(consensus,num_epoch, run, graph,ker, 
    #              save_fig, fol_name,cuda_a, x_range, x_count, l, e_sigma, C)



