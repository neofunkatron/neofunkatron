"""
Create and save percolation data for standard, brain, and model graphs.
"""

import networkx as nx
import numpy as np
import os
from os import path as op
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from extract import brain_graph
from metrics import percolation as perc
from metrics import binary_undirected as und_metrics
import brain_constants as bc
from random_graph.binary_directed import source_growth
from random_graph import binary_undirected as und_graphs

from config.graph_parameters import LENGTH_SCALE, SW_REWIRE_PROB, BRAIN_SIZE

save_files = True
##############################################################################


def construct_graph_list_und(graphs_to_const):
    """Construct and return a list of graphs so graph construction is easily
    repeatable.

    Can handle: Random, Small-world, Scale-free, SGPA, SGPA-random"""

    graph_list = []

    # Always construct and add Allen Institute mouse brain to list
    G_brain = brain_graph.binary_undirected()[0]
    graph_list.append(G_brain)

    # Calculate degree & clustering coefficient distribution
    n_nodes = G_brain.order()

    brain_degree = nx.degree(G_brain).values()
    brain_degree_mean = np.mean(brain_degree)

    # Construct degree controlled random
    if 'Random' in graphs_to_const:
        G_RAND = und_graphs.random_simple_deg_seq(
            sequence=brain_degree, brain_size=BRAIN_SIZE, tries=1000)[0]
        graph_list.append(G_RAND)

    # Construct small-world graph
    if 'Small-world' in graphs_to_const:
        graph_list.append(nx.watts_strogatz_graph(
            n_nodes, int(round(brain_degree_mean)), SW_REWIRE_PROB))

    # Construct scale-free graph
    if 'Scale-free' in graphs_to_const:
        graph_list.append(nx.barabasi_albert_graph(
            n_nodes, int(round(brain_degree_mean / 2.))))

    # Construct SGPA graph
    if 'SGPA' in graphs_to_const:
        G_SGPA = source_growth(bc.num_brain_nodes, bc.num_brain_edges_directed,
                               L=LENGTH_SCALE)[0]
        graph_list.append(G_SGPA.to_undirected())

        # Construct degree-controlled SGPA graph
        if 'SGPA-random' in graphs_to_const:
            SGPA_degree = nx.degree(G_SGPA).values()
            G_SGPA_RAND = und_graphs.random_simple_deg_seq(
                sequence=SGPA_degree, brain_size=BRAIN_SIZE, tries=1000)[0]
            graph_list.append(G_SGPA_RAND)

    # Error check that we created correct number of graphs
    if len(graph_list) != len(graphs_to_const):
        raise RuntimeError('Graph list/names don\'t match')

    return graph_list

##################
# Construct graphs
##################

repeats = 100  # Number of times to repeat percolation
prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
thresh_list = np.arange(150, -1, -5)
node_order = 426

graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free', 'SGPA']

# Undirected functions
func_list = [(perc.lesion_met_largest_component, 'Largest component'),
             (und_metrics.global_efficiency, 'Global efficiency'),
             (und_metrics.edge_count, 'Edge count'),
             (und_metrics.density, 'Connection density')]

#################
# Do percolation
#################

print 'Building percolation data...'
print 'Graphs: ' + str(graph_names)
if not save_files and repeats > 10:
    '\nAre you sure you don\'t want to save results?\n'

# Initialize matrices to store lesioning data
targ = np.zeros((len(graph_names), len(func_list), len(lesion_list), repeats))

# Percolation
for ri in np.arange(repeats):
    print 'Lesioning; repeat ' + str(ri + 1) + ' of ' + str(repeats)
    graph_list = construct_graph_list_und(graph_names)

    # Cycle over graphs and metric functions
    for gi, G in enumerate(graph_list):
        for fi, (func, func_label) in enumerate(func_list):
            # Carry out targeted attack
            targ[gi, fi, :, ri] = perc.percolate_degree(G.copy(), lesion_list,
                                                        func)
        print ' ... Done'

###############
# Save results
###############

if save_files:
    print 'Saving data for: '
    for gi, G in enumerate(graph_list):
        print '\tSaving: ' + graph_names[gi],

        save_fname = graph_names[gi] + '_undirected_perc.pkl'

        outfile = open(save_fname, 'wb')
        pickle.dump({'graph_name': graph_names[gi],
                     'metrics_list': [f.func_name for (f, f_label) in
                                      func_list],
                     'repeats': repeats,
                     'metrics_label': [f_label for (f, f_label) in func_list],
                     'data_targ': targ[gi, :, :, :],
                     'removed_rand': prop_rm,
                     'removed_targ': lesion_list,
                     'removed_targ_thresh': thresh_list}, outfile)
        outfile.close()

        print ' ... Done'
