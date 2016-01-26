import pdb
"""
Created on Wed Aug 27 22:31:09 2014

@author: rkp

Set of functions for pulling out the nodes and edges according to specific
ranking criteria.
"""

import numpy as np
import operator
import networkx as nx


def reciprocity(W_net):
    """Calculate the percentage of reciprocal connections."""
    W_binary = W_net > 0
    np.fill_diagonal(W_binary, False)
    total_cxns = W_binary.sum()
    recip_cxns = (W_binary * W_binary.T).sum()
    arecip_cxns = total_cxns - recip_cxns


#    import pdb; pdb.set_trace()
    recip_coeff = recip_cxns / (np.sum(W_net)*2.)
    return recip_coeff


def out_in(W_net, labels, binarized=True):
    """Calculate the output/input ratio given the weight matrix."""
    if binarized:
        W = (W_net > 0).astype(float)
    else:
        W = W_net.copy()
    # Calculate total output & input connections
    out_total = W.sum(axis=1)
    in_total = W.sum(axis=0)
    out_in_vec = out_total.astype(float) / in_total
    # Put into dictionary format
    out_dict = {labels[idx]: out_total[idx] for idx in range(len(labels))}
    in_dict = {labels[idx]: in_total[idx] for idx in range(len(labels))}
    out_in_dict = {labels[idx]: out_in_vec[idx] for idx in range(len(labels))}

    return out_dict, in_dict, out_in_dict


def get_ranked(criteria_dict, high_to_low=True):
    """Get labels & criteria, sorted (ranked) by criteria."""

    dict_list_sorted = sorted(criteria_dict.iteritems(),
                              key=operator.itemgetter(1), reverse=high_to_low)

    labels_sorted = [item[0] for item in dict_list_sorted]
    criteria_sorted = [item[1] for item in dict_list_sorted]

    return labels_sorted, criteria_sorted


def node_edge_overlap(node_list, edge_list):
    """Calculate the overlap of a set of nodes and edges.
    Returns which edges are touching a node and which connect two nodes."""

    # Calculate how many edges contain at least one node in node list
    edges_touching = [edge for edge in edge_list if edge[0] in node_list
                      or edge[1] in node_list]
    edges_connecting = [edge for edge in edge_list if edge[0] in node_list
                        and edge[1] in node_list]

    return edges_touching, edges_connecting


def bidirectional_metrics(W_net, coords, labels, binarized=False):
    """Calculate bidirectionality metrics for a graph given its weights.

    Returns:
        List of labeled nonzero edges, (Ne x 3) array of distance,
        bidirectionality coefficient, and connection strength."""
    if binarized:
        W_bi = (W_net > 0).astype(float)
    else:
        W_bi = W_net.copy()

    # Get nonzero elements of W_bi
    nz_idxs = np.array(W_bi.nonzero()).T
    nz_idxs = np.array([nz_idx for nz_idx in nz_idxs
                        if labels[nz_idx[0]][:-2] != labels[nz_idx[1]][:-2]])

    # Generate edge labels
    edges = [(labels[nz_idx[0]], labels[nz_idx[1]]) for nz_idx in nz_idxs]

    # Make array for storing bidirectional metrics
    bd_metrics = np.zeros((len(edges), 3), dtype=float)

    # Calculate all metrics
    for e_idx, nz in enumerate(nz_idxs):
        # Distance
        d = np.sqrt(np.sum((coords[nz[0], :] - coords[nz[1], :]) ** 2))
        # Strength
        s = W_bi[nz[0], nz[1]] + W_bi[nz[1], nz[0]]
        # Bidirectionality coefficient
        bdc = 1 - np.abs(W_bi[nz[0], nz[1]] - W_bi[nz[1], nz[0]]) / s
        # Store metrics
        bd_metrics[e_idx, :] = [d, bdc, s]

    return edges, bd_metrics


def whole_graph_metrics(graph, weighted=False):
    graph_metrics = {}

    # Shortest average path length
    graph_metrics['avg_shortest_path'] = \
        nx.average_shortest_path_length(graph, weight=weighted)

    # Average eccentricity
    ecc_dict = nx.eccentricity(graph)
    graph_metrics['avg_eccentricity'] = np.mean(np.array(ecc_dict.values()))

    # Average clustering coefficient
    # NOTE: Option to include or exclude zeros
    graph_metrics['avg_ccoeff'] = \
        nx.average_clustering(graph, weight=weighted, count_zeros=True)

    # Average node betweeness
    avg_node_btwn_dict = nx.betweenness_centrality(graph, normalized=True)
    graph_metrics['avg_node_btwn'] = \
        np.mean(np.array(avg_node_btwn_dict.values()))

    # Average edge betweeness
    avg_edge_btwn_dict = nx.edge_betweenness_centrality(graph, normalized=True)
    graph_metrics['avg_edge_btwn'] = \
        np.mean(np.array(avg_edge_btwn_dict.values()))

    # Number of isolates
    graph_metrics['isolates'] = len(nx.isolates(graph))

    return graph_metrics
