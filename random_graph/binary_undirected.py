"""
Created on Wed Nov 12 11:18:14 2014

@author: rkp, wronk

Functions for generating random binary undirected graphs not included in
NetworkX.
"""

import numpy as np
import networkx as nx
import graph_tools.auxiliary as aux_tools


def ER_distance(N=426, p=.086, brain_size=[7., 7., 7.]):
    """Create an Erdos-Renyi random graph in which each node is assigned a
    position in space, so that relative positions are represented by a distance
    matrix."""
    # Make graph & get adjacency matrix
    G = nx.erdos_renyi_graph(N, p)
    A = nx.adjacency_matrix(G)
    # Randomly distribute nodes in space & compute distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    return G, A, D


def pure_geometric(N, N_edges, L, brain_size=[7., 7., 7.]):
    """
    Create a pure geometric model in which nodes are embedded in 3-D space and connect preferentially to
    physically nearby nodes.

    :param N: number of nodes
    :param N_edges: number of edges
    :param L: length constant
    :param brain_size: volume in which to distribute nodes in
    :return: graph, adjacency matrix, distance matrix
    """
    # randomly distribute nodes in space & compute distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    # make an adjacency matrix and graph
    A = np.zeros((N, N), dtype=float)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # initialize some helper variables
    node_idxs = np.arange(N, dtype=int)
    degs = np.zeros((N, ), dtype=int)
    fully_connected = np.zeros((N, ), dtype=bool)

    # add edges
    edge_ctr = 0
    while edge_ctr < N_edges:

        # pick source randomly
        src = np.random.choice(node_idxs[fully_connected == False])

        # get distance-dependent probabilities to all available targets
        d_probs = np.exp(-D[src, :]/L)  # unnormalized

        # compute which nodes are available to connect to
        unavailable_targs = fully_connected.copy()  # no fully connected nodes
        unavailable_targs[src] = True  # can't connect to self
        unavailable_targs[A[src, :] == 1] = True  # can't connect to already connected nodes

        # set probability to zero if node unavailable
        d_probs[unavailable_targs] = 0.
        # normalize
        d_probs /= d_probs.sum()

        # randomly pick a target
        targ = np.random.choice(node_idxs, p=d_probs)

        # add edge to graph and adjacency matrix
        G.add_edge(src, targ)
        A[src, targ] = 1
        A[targ, src] = 1

        # update degrees
        degs[src] += 1
        degs[targ] += 1

        # update available_from
        if degs[src] == N:
            fully_connected[src] = False
        if degs[targ] == N:
            fully_connected[targ] = False

        edge_ctr += 1

    return G, A, D


def random_simple_deg_seq(sequence, brain_size=[7., 7., 7.], seed=None,
                          tries=10):
    '''Wrapper function to get a SIMPLE (no parallel or self-loop edges) graph
    that has a given degree sequence.

    This graph is used conventionally as a control because it yields a random
    graph that accounts for degree distribution.

    Parameters:
        sequence: list of int
            Degree of each node to be added to the graph.
        brain_size: list of 3 floats
            Size of the brain to use when distributing  node locations.
            Added for convenience, but does not affect connectivity pattern.
        seed: hashable object for random seed
            Seed for the random number generator.
        tries: int
            Number of attempts at creating graph (function will retry if
            self-loops exist.
    Returns:
        Networkx graph object, adjacency matrix, and random distances'''

    G = nx.random_degree_sequence_graph(sequence=sequence, seed=seed,
                                        tries=tries)

    A = nx.adjacency_matrix(G)
    N = len(sequence)
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    return G, A, D
