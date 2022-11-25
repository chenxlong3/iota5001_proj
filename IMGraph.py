import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import time, os
from math import comb
from random import uniform, seed, choice
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from utils import *

class IMGraph:
    def __init__(
        self,
        file_path: str, 
        p:float = .5,
        mc:int = 1000,
        eps:float = 0.2,
        l:int = 1,
        max_k = 5,
        k_step = 1
    ) -> None:
        self.file_path = file_path
        self.p = p
        self.mc = mc
        self.method_spread_map = {}
        self.method_seed_map = {}
        self.method_time_map = {}
        self.method_seed_idx = {}
        self.k = max_k
        self.eps = eps
        self.l = l
        self.k_list = [i for i in range(1, max_k+1, k_step)]
        try:
            self.G_nx = self.load_G_nx()
            
        except Exception as e:
            print("Failed to load the graph by networkx")
            print("Error:")
            print(e)
        try:
            self.G = ig.Graph.from_networkx(self.G_nx)
        except Exception as e:
            print("Failed to load the graph by networkx")
            print("Error:")
            print(e)
        self.n = self.G.vcount()
        self.m = self.G.ecount()
        return
    
    def test(self) -> None:
        print(self.file_path)
    
    def set_p(self, p) -> None:
        self.p = p
        return
    
    def load_G_nx(self) -> nx.classes.graph.Graph:
        if self.file_path.endswith("gml"):
            return nx.read_gml(self.file_path)
        elif self.file_path.endswith("mtx"):
            return read_mtx(self.file_path)
        elif self.file_path.endswith("edges"):
            try:
                return nx.read_edgelist(self.file_path)
            except:
                return nx.read_edgelist(self.file_path, data=(("weight", float),))
        print("Cannot process such a file format")
        return None

    # Independent cascade model
    # Used to compute influence spread
    def IC(self, S) -> float:
        """
        Input:
            G: igraph Graph
            S: seed set
            p: probability threshold
            mc: number of MC simulations
        Output:
            average number of influenced nodes
        """
        spread = []     # The number of influenced nodes, starting from S
        # Loop for MC simulations
        for i in range(self.mc):
            # new_active: Newly activated nodes
            # A: all activated nodes
            new_active, A = S[:], S[:]

            # while there are newly added nodes
            while len(new_active) > 0:
                new_ones = []
                # For every newly activated nodes
                for node in new_active:
                    # Determine neighbors that become influenced
                    np.random.seed(i)       # set random seed
                    # sampling
                    success = np.random.uniform(0,1,len(self.G.neighbors(node,mode="out"))) < self.p
                    # newly activated nodes
                    new_ones += list(np.extract(success, self.G.neighbors(node,mode="out")))
                # compute the newly activated nodes
                new_active = list(set(new_ones) - set(A))
                
                # Add newly activated nodes to the set of activated nodes
                A += new_active
            # number of all activated nodes in this instance
            # print(i, len(A))
            spread.append(len(A))
        return np.mean(spread)
    
    def brute_force(self) -> None:
        SPREAD = []
        for k in tqdm(range(1, self.k+1)):
            combs = combinations(range(self.n), k)
            max_spread = 0
            max_seeds = []
            for c in tqdm(combs):
                cur_res = self.IC(list(c))
                if cur_res > max_spread:
                    max_spread = cur_res
                    max_seeds = c
            SPREAD.append(max_spread)
        self.method_spread_map["EXACT"] = SPREAD
        self.method_seed_idx["EXACT"] = max_seeds

    def proxy(self, proxy="pagerank") -> None:
        st_time = time.time()
        
        Q = zip(range(self.G.vcount()), getattr(self.G, proxy)())
        Q = sorted(Q, key = lambda x: x[1], reverse=True)

        SEED = [Q[i][0] for i in range(self.k)]

        self.method_time_map[proxy] = [time.time() - st_time]*len(self.k_list)

        spread = [self.IC(SEED[:i]) for i in self.k_list]
        self.method_spread_map[proxy] = spread
        self.method_seed_idx[proxy] = SEED
        self.method_seed_map[proxy] = [self.G.vs[idx]["_nx_name"] for idx in SEED]
        return
    
    def run_proxy_methods(self) -> None:
        for metric in proxy_metrics:
            self.proxy(metric)
        return
    # Greedy Algorithm
    def run_greedy(self) -> None:
        """
        Input:
            G: igraph Graph
            k: size of seed set
            p: threshold
            mc: number of mc simulation
        Output:
            S: solution seed set
            spread: number of influenced vertices
        """
        SEED, spread, timelapse, start_time = [], [], [], time.time()
        # loop for k nodes selection
        for _ in tqdm(range(self.k)):
            best_spread = 0    # initialization 
            # for every node that is not in S
            for j in set(range(self.G.vcount())) - set(SEED):
                s = self.IC(SEED+[j])
                if s > best_spread:
                    best_spread, node = s, j
            SEED.append(node)

            # Estimated spread and elapsed time
            spread.append(best_spread)
            timelapse.append(time.time() - start_time)
        self.method_spread_map["greedy"] = spread
        self.method_time_map["greedy"] = timelapse
        self.method_seed_idx["greedy"] = SEED
        self.method_seed_map["greedy"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]
        return
    
    # Cost Effective Lazy Forward
    def run_celf(self):
        st_time = time.time()       # start time
        # marginal gain for every node
        # spread from every single node
        marg_gain = [self.IC([node]) for node in tqdm(range(self.n))]
        # sort the nodes by marginal gain
        Q = sorted(zip(range(self.n), marg_gain), key=lambda x: x[1], reverse=True)

        # seed set initialization: the first node
        # spread: number of all influenced nodes
        # SPREAD: # influenced nodes list
        S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
        Q, LOOKUPS, timelapse = Q[1:], [self.n], [time.time() - st_time]

        
        for _ in tqdm(range(self.k-1)):
            checked, node_lookup = False, 0
            # till the node with the highest MG does not change
            while not checked:
                node_lookup += 1    # The number of times the spread is computed
                current = Q[0][0]
                # calculate the MG of the current node
                Q[0] = (current, self.IC(S + [current]) - spread)
                Q = sorted(Q, key=lambda x: x[1], reverse=True)

                # if new MG is still the highest, exit the loop
                if Q[0][0] == current:
                    checked = True
                
            spread += Q[0][1]
            S.append(Q[0][0])
            SPREAD.append(spread)
            LOOKUPS.append(node_lookup)
            timelapse.append(time.time()-st_time)

            Q = Q[1:]
        self.method_spread_map["CELF"] = SPREAD
        self.method_time_map["CELF"] = timelapse
        self.method_seed_idx["CELF"] = S
        self.method_seed_map["CELF"] = [self.G.vs[idx]["_nx_name"] for idx in S]
        return
    
    def get_RRS(self):
        """
        Inputs:
            G: igraph Graph
            p: Propagation probability
        """
        source = choice(self.G.vs.indices)
        # mask = np.random.uniform(0, 1, len(self.G.neighbors(source,mode="out"))) < self.p
        samp_G = np.array(self.G.get_edgelist())[np.random.uniform(0, 1, self.m) < self.p]

        new_nodes, RRS0 = [source], [source]
        while new_nodes:
            tmp = [edge for edge in samp_G if edge[1] in new_nodes]
            tmp = [edge[0] for edge in tmp]
            RRS = list(set(RRS0+tmp))

            new_nodes = list(set(RRS) - set(RRS0))  # New nodes in the RR set

            RRS0 = RRS
        return RRS
    
    def run_RIS(self) -> None:
        st_time = time.time()
        R = [self.get_RRS() for _ in range(self.mc)]

        SEED, timelapse = [], []

        for _ in range(self.k):
            flat_list = [item for sublist in R for item in sublist]
            seed = Counter(flat_list).most_common()[0][0]
            SEED.append(seed)
            R = [rrs for rrs in R if seed not in rrs]
            timelapse.append(time.time() - st_time)
        
        # self.method_spread_map["RIS"] = SPREAD
        self.method_time_map["RIS"] = timelapse
        self.method_seed_idx["RIS"] = SEED
        self.method_seed_map["RIS"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]
        return
    def run_TIM(self):
        n, k, eps, l = self.n, self.k, self.eps, self.l
        
        st_time = time.time()
        kpt = self.kpt_estimation()
        lam = (8+2*self.eps)*self.n*(l*np.log(n) + np.log(comb(n, k)) + np.log(2))*np.power(eps, -2)
        theta = int(np.ceil(lam / kpt))
        print(theta)
        R = [self.get_RRS() for _ in range(theta)]
        SEED, timelapse = [], []

        for _ in range(self.k):
            flat_list = [item for sublist in R for item in sublist]
            seed = Counter(flat_list).most_common()[0][0]
            SEED.append(seed)
            R = [rrs for rrs in R if seed not in rrs]
            timelapse.append(time.time() - st_time)
        self.method_time_map["TIM"] = timelapse
        self.method_seed_idx["TIM"] = SEED
        self.method_seed_map["TIM"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]
    
    def get_w(self, R):
        res = 0
        for node in R:
            self.G.indegree()[node]
        return res
            

    def kpt_estimation(self):
        for i in range(1, int(np.ceil(np.log2(self.n)-1))):
            ci = (6*self.l*np.log(self.n) + 6*np.log(np.log2(self.n))) * np.power(2, i)
            _sum = 0
            for j in range(1, int(np.ceil(ci))):
                R = self.get_RRS()
                kappa = 1 - np.power(1 - self.get_w(R)/self.m, self.k)
                _sum += kappa
            if _sum / ci > 1 / np.power(2, i):
                return self.n*_sum/(2*ci)
            return 1

    
    def estimate_spread(self, method:str) -> None:
        SPREAD = []
        S = self.method_seed_idx[method]
        for i in range(len(S)):
            SPREAD.append(self.IC(S[:i+1]))
        self.method_spread_map[method] = SPREAD
        return
    
    def run_all_methods(self) -> None:
        self.run_proxy_methods()
        self.run_greedy()
        self.run_celf()
        self.run_RIS()
        self.estimate_spread("RIS")
        return
