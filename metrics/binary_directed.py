"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for binary undirected graphs.
"""
from __future__ import division
import numpy as np
import networkx as nx
from scipy import stats


def reciprocity(G):
    """Calculate the reciprocity coefficient of a directed graph.

    Reciprocity is defined as the number of bidirectional links over the total number of links"""

    return len(G.to_undirected(reciprocal=True).edges()) / len(G.to_undirected(reciprocal=False).edges())


def efficiency_matrix(G):
    """Calculate the efficiency (the inverse of the length of the shortest
    directed path) for all pairs of nodes in a graph.  Rows correspond to
    source nodes, columns to target nodes.

    Parameters
    ----------
    G : networkX graph

    Returns
    -------
    efficiency : ndarray, shape(G.number_of_nodes(), G.number_of_nodes()
        Matrix containing pairwise efficiency calculations"""

    shortest_path_lengths = nx.shortest_path_length(G)

    efficiency = np.zeros((len(G.nodes()), len(G.nodes())), dtype=float)

    for src in shortest_path_lengths.keys():
        for targ in shortest_path_lengths[src].keys():
            if src is not targ:
                if nx.has_path(G, src, targ):
                    efficiency[src, targ] = 1. / shortest_path_lengths[src][targ]
                else:
                    efficiency[src, targ] = 0.

    return efficiency


def power_law_fit_deg_cc(G):
    """
    Fit a power-law model to a graphs clustering (y-axis) vs. degree (x-axis) scatter plot.
    Fits a line to a log-log plot. Return values are relative to log-log plot.
    :param G: graph
    :return: slope, intercept, r-value, p-value, stderr
    """
    deg = np.array(nx.degree(G.to_undirected()).values())
    cc = np.array(nx.clustering(G.to_undirected()).values())

    # remove zero-valued items, as they'll mess up fitting to log plot
    mask = (deg > 0) * (cc > 0)
    deg = deg[mask]
    cc = cc[mask]

    return stats.linregress(np.log(deg), np.log(cc))


def dists_by_reciprocity(a, d):
    """
    Given an adjacency and distance matrix, return arrays of distances corresponding to
    reciprocal and nonreciprocal edges. Self-connections are ignored.
    :param a: adjacency matrix
    :param d: distance matrix
    :return: array of distances of reciprocal edges, array of distances of nonreciprocal edges
    """
    r_mask = (a & a.T).astype(bool)
    self_mask = ~np.eye(len(a), dtype=bool)
    r_dists = d[r_mask & self_mask].flatten()
    non_r_dists = d[a.astype(bool) & (~r_mask) & self_mask].flatten()

    return r_dists, non_r_dists