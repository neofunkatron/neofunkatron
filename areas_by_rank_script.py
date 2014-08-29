import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: rkp

Analyze properties of specific brain areas with extreme ranks according to
specific criteria.
"""

import pprint as pp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import collect_areas
import network_gen
import area_compute
import network_viz

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

# Set relative directory path to linear model & ontology
dir_LM = '../friday-harbor/linear_model'

# Load weights & p-values
W, P, row_labels, col_labels = network_gen.load_weights(dir_LM)
# Threshold weights according to weights & p-values
W_net, mask = network_gen.threshold(W, P, p_th=p_th, w_th=w_th)
# Set weights to zero if they don't satisfy threshold criteria
W_net[W_net == -1] = 0.
# Set diagonal weights to zero
np.fill_diagonal(W_net, 0)
# Put everything in a dictionary
W_net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
              'data': W_net}
# Convert to networkx graph object
G = network_gen.import_weights_to_graph(W_net_dict)

# Collect & sort areas & edges according to various attributes
sorted_areas = collect_areas.collect_and_sort(G,W_net,labels=row_labels,
                                              print_out=True)

# Compute feature dictionary for all areas
area_dict = area_compute.get_feature_dicts(G.nodes(),G,W_net,row_labels)

# Visualize individual areas & their cxns
num_top_deg = 3
for top_deg_idx in range(num_top_deg):
    # Get pair of areas
    area0 = sorted_areas['degree_labels'][2*top_deg_idx]
    area1 = sorted_areas['degree_labels'][2*top_deg_idx+1] # Get neighbors for each area
    neighbors0 = area_dict[area0]['neighbors']
    neighbors1 = area_dict[area1]['neighbors']
    # Get edges for each area
    edges0 = [(area0,areaX) for areaX in neighbors0]
    edges1 = [(area1,areaX) for areaX in neighbors1]
    # Put areas and neighbors together & remove duplicates
    all_nodes = [area0,area1] + neighbors0 + neighbors1
    all_edges = edges0 + edges1
    all_nodes = list(np.unique(all_nodes))
    all_edges = list(np.unique(all_edges))
    # Get centroids for nodes
    all_centroids = [area_dict[area]['centroid'] for area in all_nodes]
    all_centroids = np.array(all_centroids)
    # Swap columns so that S <-> I is on z axis
    all_centroids = all_centroids.take([0, 2, 1], 1)
    # Get logical indices of area nodes
    area_nodes = np.array([name in [area0, area1] for name in all_nodes])
    node_label_set = area_nodes
    edge_label_set = np.zeros((len(all_edges),),dtype=bool)
    # Specify sizes
    node_sizes = 20*np.ones((len(all_nodes),))
    node_sizes[area_nodes] = 100
    node_colors = np.array(['k' for node_idx in range(len(all_nodes))])
    node_colors[area_nodes] = 'r'
    # Plot 3D nodes
    network_viz.plot_3D_network(node_names=all_nodes,
                                node_positions=all_centroids,
                                node_label_set=node_label_set,
                                edges=all_edges,
                                edge_label_set=edge_label_set,
                                node_sizes=node_sizes,
                                node_colors=node_colors)
