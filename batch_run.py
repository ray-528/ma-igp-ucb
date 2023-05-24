""" 
Distributed Optimization via Kernelized Multi-armed Bandits
"""

import argparse
from Main import bandit_ma_gp
import numpy as np
import random

# save_fig = 'No'
save_fig = 'yes'
fol_name = "plots"

#Hyper-parameters
x_range = [0,1]
x_count = 100
e_sigma = 0.2
l = 0.1
C = 2 # number of iterations in a stage

parser = argparse.ArgumentParser()
parser.add_argument('--con', type=str,  help='Type of consensus')
parser.add_argument('--epo', type=str,  help='Number of epochs')
parser.add_argument('--ker', type=str,  help='Kernel')
parser.add_argument('--net', type=str,  help='Network graph')
parser.add_argument('--run', type=str,  help='Which run it is')
args = parser.parse_args()
consensus = args.con
graph = args.net
ker = args.ker
run = int(args.run) # Experiment number to study the mean behaviour
seed = 500 + run
np.random.seed(seed)
random.seed(seed)
num_epoch = int(args.epo) # Number of runs in single episode
cuda_a = "yes" # Is cuda available

bandit_ma_gp(consensus,num_epoch, run, graph,ker, 
                 save_fig, fol_name,cuda_a, x_range, x_count, l, e_sigma, C)

