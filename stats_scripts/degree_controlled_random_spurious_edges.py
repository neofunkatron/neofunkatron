"""
Created on Wed Jan 27th 2016

@author: rkp
degree_controlled_random_spurious_edges.py

Calculate how many self/multi-edges are removed on average when making a
directed degree-controlled random graph with degree-distribution fixed to the brain's.
"""
from __future__ import division, print_function
import numpy as np
import networkx as nx
from extract import brain_graph
from random_graph import binary_directed


N_GRAPHS = 30


G_brain_dir, _, _ = brain_graph.binary_directed()

in_degree_seq = G_brain_dir.in_degree().values()
out_degree_seq = G_brain_dir.out_degree().values()

print('Making directed degree-controlled random graphs')

n_edges_lost_dir = []
for ctr in range(N_GRAPHS):
    print(ctr)
    G_dir, _, _ = binary_directed.random_directed_deg_seq(in_degree_seq, out_degree_seq, simplify=True)
    n_edges_lost_dir.append(len(G_brain_dir.edges()) - len(G_dir.edges()))

print('Directed: mean edges lost +- std = {} +- {}'.format(
    np.mean(n_edges_lost_dir), np.std(n_edges_lost_dir))
)