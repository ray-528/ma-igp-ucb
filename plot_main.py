#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_group_reward(consensus,num_epoch,avg_rew_over_runs,
                      save_fig, fol_name, graph,ker, num_ag):
    # Plotting average reward
    
    mean_avg_rew = np.mean(avg_rew_over_runs,1)
    # std_avg_rew = np.std(avg_rew_over_runs,1)
    
    time = np.linspace(1,num_epoch,num_epoch)
    fig, ax = plt.subplots()
    # ax.fill_between(time, mean_avg_rew - std_avg_rew,
    #         mean_avg_rew + std_avg_rew, color ='b', alpha = 0.15)
    plt.plot(time, mean_avg_rew, color ='b', label = "Avg. Reward")
    plt.legend()
    plt.title('Group reward at each epoch')
    plt.ylabel('Network reward')
    plt.xlabel('T')
    if save_fig =="yes":
        plt.savefig(f"{fol_name}/{consensus}_{graph}-graph_{ker}_{num_ag}ag_reward.pdf")
    else:
        plt.show()

def plot_cumu_regret(consensus, num_epoch,cumu_regret_over_runs,
                     save_fig, fol_name,graph,ker, num_ag):
    mean_cumu_regret = np.mean(cumu_regret_over_runs,1)/num_ag
    # std_cumu_regret = np.std(cumu_regret_over_runs,1)
    
    time = np.linspace(1,num_epoch,num_epoch)
    fig, ax = plt.subplots()
    # ax.fill_between(time, mean_cumu_regret - std_cumu_regret,
    #         mean_cumu_regret + std_cumu_regret, color ='lightgray', alpha = 0.5)
    plt.plot(time, mean_cumu_regret, color ='b', label = "Cumu. regret")
    plt.legend()
    plt.title(consensus +': Cumulative Regret')
    plt.ylabel('R(T)')
    plt.xlabel('T')
    if save_fig =="yes":
        plt.savefig(f"{fol_name}/{consensus}_{graph}-graph_{ker}_{num_ag}ag_cum-regret.pdf")
    else:
        plt.show()
    
        
        