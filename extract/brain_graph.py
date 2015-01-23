import pdb
"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Functions to extract graphs from Allen mouse connectivity.
"""

LINEAR_MODEL_DIRECTORY = '../../mouse_connectivity_data/linear_model'
STRUCTURE_DIRECTORY = '../../mouse_connectivity_data'

import auxiliary as aux
import graph_tools.auxiliary as aux_tools
import networkx as nx
import numpy as np


def binary_undirected(p_th=.01, w_th=0, data_dir=LINEAR_MODEL_DIRECTORY):
    """Load brain as binary undirected graph.
    
    Returns:
        NetworkX graph, adjacency matrix, row labels & column labels"""
    # Load weights & p-values
    W, P, labels = aux.load_W_and_P(data_dir)
    # Threshold weights via weights & p-values
    W[(W < w_th)] = 0.
    W[(P > p_th)] = 0.
    # Symmetrize W by summing reciprocal weights
    W = W + W.T
    # Set self-weights to zero
    np.fill_diagonal(W,0.)
    # Create adjacency matrix
    A = (W > 0).astype(int)
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_matrix(A)
    
    return G, A, labels
    
    
def binary_directed(p_th=.01, w_th=0, data_dir=LINEAR_MODEL_DIRECTORY):
    """Load brain as binary directed graph.
    
    Returns:
        NetworkX graph, adjacency matrix, row labels & column labels"""
    # Load weights & p-values
    W, P, labels = aux.load_W_and_P(data_dir)
    # Threshold weights via weights & p-values
    W[(W < w_th)] = 0.
    W[(P > p_th)] = 0.

    # Set self-weights to zero
    np.fill_diagonal(W,0.)
    
    # Create adjacency matrix
    A = (W > 0).astype(int)
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    
    return G, A, labels
    
    
def weighted_undirected(p_th=.01, w_th=0, data_dir=LINEAR_MODEL_DIRECTORY):
    """Load brain as binary undirected graph.
    
    Returns:
        NetworkX graph, weight matrix, row labels & column labels"""
    # Load weights & p-values
    W, P, labels = aux.load_W_and_P(data_dir=data_dir)

    # Threshold weights via weights & p-values
    W[(W < w_th)] = 0.
    W[(P > p_th)] = 0.
    # Symmetrize W by summing reciprocal weights
    W = W + W.T
    
    # Set self-weights to zero
    np.fill_diagonal(W,0.)
    
    # Create graph from weight matrix
    G = nx.from_numpy_matrix(W)
    
    return G, W, labels


def distance_matrix(lm_dir=LINEAR_MODEL_DIRECTORY, cent_dir=STRUCTURE_DIRECTORY,
                    in_mm=True):
    """Compute distance matrix from centroid data.
    
    Args:
        lm_dir: Directory containing linear model data
        cent_dir: Directory containing centroid data
        in_mm: Set to true to return dist matrix in mm instead of 100um units
    Returns:
        distance matrix, centroid matrix"""
    # Get labels
    _, _, labels = aux.load_W_and_P(data_dir=lm_dir)
    # Load centroids
    centroids = aux.load_centroids(labels, data_dir=cent_dir, in_mm=in_mm)
    # Compute distance matrix
    dist_mat = aux_tools.dist_mat(centroids)
    
    return dist_mat, centroids