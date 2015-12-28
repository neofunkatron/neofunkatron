import pdb
"""
Created on Mon Sep  1 22:37:27 2014

@author: rkp

Code to generate a scale-free graph with adjustable clustering coefficient
according to the paper Herrera & Zufiria 2011.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io.matlab as scio


def dist_mat(centroids):
    """Compute a distance matrix from 3D centroids."""
    
    D = np.zeros((centroids.shape[0],centroids.shape[0]),dtype=float)
    for r_idx in range(D.shape[0]):
        for c_idx in range(D.shape[1]):
            d = np.sqrt(np.sum((centroids[r_idx,:] - centroids[c_idx])**2))
            D[r_idx,c_idx] = d
    return D


def get_coords():
    import os
    
    data_dir = os.getenv('DBW_DATA_DIRECTORY')
    A = scio.loadmat(data_dir+'/Structure_coordinates_and_distances.mat')
    contra_coords = A['cntrltrl_coordnt_dctnry']
    contra_type = contra_coords.dtype
    contra_names = contra_type.names
    contra_dict = {contra_names[k] + '_L':contra_coords[0][0][k][0] for k in range(len(contra_coords[0][0]))}

    ipsi_coords = A['ipsltrl_coordnt_dctnry']
    ipsi_type = ipsi_coords.dtype
    ipsi_names = ipsi_type.names
    ipsi_dict = {ipsi_names[k] + '_R':ipsi_coords[0][0][k][0] for k in range(len(ipsi_coords[0][0]))}

    contra_dict.update(ipsi_dict)
    
    return contra_dict
    
    
def get_distances():
    # Might want to compute your own Euclidean distances here?
    A = get_coords()
    names = A.keys()
    D = np.zeros([len(A),len(A)])
    for i in range(len(A)):
	current_i = names[i]
	for j in range(len(A)):
	    current_j = names[j]
	   
	    D[i,j] = sum((A[current_i] - A[current_j])**2)
	   
    return D,names
    
def get_edge_distances(G):
    edges = G.edges()
    distances = {}
    for edge in edges:
	centroid1 = G.node[edge[0]]
	centroid2 = G.node[edge[1]]
	d = sum((centroid1-centroid2)**2)
	distances[edge] = d
	
    return distances
    
def compute_average_distance(G):
    distances = get_edge_distances(G)
    avg_distance = {}
    for node in G.nodes():
	current_distances = [distances[edge] for edge in distances if node in edge]
	avg_distance[node] = np.mean(current_distances)
	
    return avg_distance
    
if __name__ == '__main__':
    G = scale_free_cc_graph(n=426,m=12,k0=3,p=np.array([1]),fp=np.array([1]))
