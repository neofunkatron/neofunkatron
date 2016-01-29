# -*- coding: utf-8 -*-

"""
Created Sat May 23 14:11:50 2015

@author: wronk

Create a table of useful metrics for undirected standard graphs and model
"""

import os
import os.path as op
import numpy as np
import networkx as nx
import csv
import pickle

import extract.brain_graph
from random_graph import binary_undirected
from random_graph.binary_directed import source_growth
from metrics import binary_undirected as und_metrics
import brain_constants as bc
from config.graph_parameters import LENGTH_SCALE, SW_REWIRE_PROB, BRAIN_SIZE
reload(und_metrics)


def calc_metrics(G, metrics):
    """Helper function to calculate list of (function) metrics"""
    metric_vals = np.zeros((len(metrics)))
    for fi, func in enumerate(metrics):
        metric_vals[fi] = func(G)

    return metric_vals

###############################
# Parameters
###############################

# SET YOUR SAVE DIRECTORY
save_dir = os.environ['DBW_SAVE_CACHE']

repeats = 1

# Set the graphs and metrics you wisht to include
graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free',
               'SGPA']
metrics = [nx.average_clustering, nx.average_shortest_path_length,
           und_metrics.global_efficiency, und_metrics.local_efficiency]

###############################
# Create graph/ compute metrics
###############################
# TODO: consider using a dict instead of checking for each graph type

# Initialize matrix to store metric values
met_arr = -1 * np.ones((len(graph_names), repeats, len(metrics)))

# Load mouse connectivity graph
G_brain, _, _ = extract.brain_graph.binary_undirected()
n_nodes = G_brain.number_of_nodes()
n_edges = G_brain.number_of_edges()
G_brain_copy = G_brain.copy()

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Calculate metrics specially because no repeats can be done on brain
brain_metrics = calc_metrics(G_brain, metrics)
for met_i, bm in enumerate(metrics):
    met_arr[graph_names.index('Connectome'), :, met_i] = bm(G_brain)

print 'Running metric table with %d repeats\n' % repeats
for rep in np.arange(repeats):
    # SGPA model
    if 'SGPA' in graph_names:
        G_SGPA = source_growth(bc.num_brain_nodes, L=LENGTH_SCALE)[0]
        met_arr[graph_names.index('SGPA'), rep, :] = \
            calc_metrics(G_SGPA.to_undirected(), metrics)

    # Random Configuration model (random with fixed degree sequence)
    if 'Random' in graph_names:
        G_CM = binary_undirected.random_simple_deg_seq(
            sequence=brain_degree, brain_size=BRAIN_SIZE, tries=100)[0]
        met_arr[graph_names.index('Random'), rep, :] = \
            calc_metrics(G_CM, metrics)

    # Small-world (Watts-Strogatz) model with standard reconnection prob
    if 'Small-world' in graph_names:
        G_SW = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)),
                                       SW_REWIRE_PROB)
        met_arr[graph_names.index('Small-world'), rep, :] = \
            calc_metrics(G_SW, metrics)

    # Scale-free (Barabasi-Albert) graph
    if 'Scale-free' in graph_names:
        G_SF = nx.barabasi_albert_graph(n_nodes,
                                        int(round(brain_degree_mean / 2.)))
        met_arr[graph_names.index('Scale-free'), rep, :] = \
            calc_metrics(G_SF, metrics)

    print 'Completed repeat: ' + str(rep)

# Calculate the standard deviation
std_arr = np.std(met_arr, axis=1)

##########################
# Save metrics
##########################
save_dict = dict(met_arr=met_arr, std_arr=std_arr, metrics=metrics,
                 graph_names=graph_names)

# Save original array/information and csv version averaged across repeats
f_name = op.join(save_dir, 'undirected_metrics')

# Save all original info in pickled dict (for potential plotting later)
outfile = open(f_name + '.pkl', 'wb')
pickle.dump(save_dict, outfile)
outfile.close()

# Save csvs for easily dumping data to table
met_mean = met_arr.mean(1)
csv_out = []
for mean, std in zip(met_mean, std_arr):
    csv_out.append(['%10.3f +/- %10.4f' % (m, s) for m, s in zip(mean, std)])

# Open and write csv file. Contains mean and std now
with open(f_name + '_mean_std.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(csv_out)
    csv_file.close()
