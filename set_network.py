#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selecting the multi-agent network
"""
import numpy as np
import networkx as nx
from scipy.io import loadmat

class set_net():
    def __init__(self):
        self.fname = "poplular_adj_100agents_100runs.mat"
    
    def small(self):
        num = 5
        G = nx.Graph()
        elist = [(1, 2), (2, 3), (4, 5), (4, 1),(3,1)]
        G.add_edges_from(elist)
        Adj = nx.adjacency_matrix(G).toarray()
        weight = self.get_weight(num,Adj)
        return weight
    
    def big_random(self,num):
        '''
        Random ER network
        num = number of agents
        p = edge probability
        '''
        p = 0.04
        G = nx.gnp_random_graph(num, p)
        while nx.is_connected(G)==False:
            G = nx.gnp_random_graph(num, p)
        Adj = nx.adjacency_matrix(G).toarray()
        weight = self.get_weight(num,Adj)
        return weight

    def big_fb(self,run):
        '''
        Facebook SNAP network
        num = number of agents
        run = sample number 
        '''
        num = 100
        mdic = loadmat(self.fname)
        Adj = mdic["fb_adj"][run]
        weight = self.get_weight(num,Adj)
        return weight

    def big_twitch(self,run):
        '''
        Twitch SNAP network
        num = number of agents
        run = sample number 
        '''
        num = 100
        mdic = loadmat(self.fname)
        Adj = mdic["tw_adj"][run]
        weight = self.get_weight(num,Adj)
        return weight
    
    def big_skitter(self,run):
        '''
        Skitter SNAP network
        num = number of agents
        run = sample number 
        '''
        num = 100
        mdic = loadmat(self.fname)
        Adj = mdic["sk_adj"][run]
        weight = self.get_weight(num,Adj)
        return weight
   
    def big_grid(self, num):
        '''
        Grid network
        num = square root of number of agents
        '''
        G = nx.grid_graph((num,num))
        Adj = nx.adjacency_matrix(G).toarray()
        weight = self.get_weight(num**2,Adj)
        return weight 
    
    def get_weight(self,num,Adj): 
        '''
        Getting weights of the adjacency matrix
        num = number of agents
        '''       
        w=np.zeros([num,num])
        D = np.sum(Adj,axis=1)
        for i in range(num):
            for j in range(num):
                if Adj[i,j]!=0:
                    w[i,j] = min([1/(1+D[i]),1/(1+D[j])])
            w[i,i] = 1- np.sum(w[i])
        return w
            
    def draw_net(self,G):
        nx.draw(G,node_size=20)
    
