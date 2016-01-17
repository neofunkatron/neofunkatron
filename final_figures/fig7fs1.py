"""
Create and plot percolation data for fig 7 supplemental figure 1
Connection density vs Global efficiency for exemplar networks
"""

import os
import os.path as op
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as cf
from extract import brain_graph
from metrics import percolation as perc
from metrics import binary_undirected as und_metrics
import brain_constants as bc
from random_graph.binary_directed import source_growth
from random_graph.binary_undirected import random_simple_deg_seq

from config.graph_parameters import LENGTH_SCALE

repeats = 1  # Number of times to repeat percolation (should be 1 here)
prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
thresh_list = np.arange(150, -1, -5)
node_order = 426
brain_size = [7., 7., 7.]

save_dir = os.environ['DBW_SAVE_CACHE']
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
        G_RAND = random_simple_deg_seq(sequence=brain_degree,
                                       brain_size=brain_size, tries=1000)[0]
        graph_list.append(G_RAND)

    # Construct small-world graph
    if 'Small-world' in graphs_to_const:
        graph_list.append(nx.watts_strogatz_graph(
            n_nodes, int(round(brain_degree_mean)), 0.23))

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
            G_SGPA_RAND = random_simple_deg_seq(
                sequence=SGPA_degree, brain_size=brain_size, tries=1000)[0]
            graph_list.append(G_SGPA_RAND)

    # Error check that we created correct number of graphs
    if len(graph_list) != len(graphs_to_const):
        raise RuntimeError('Graph list/names don\'t match')

    return graph_list

##################
# Construct graphs
##################

graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free',
               'SGPA']  # , 'SGPA-random']

# Undirected
func_names = ['Largest component', 'Global efficiency', 'Connection density']
func_calls = [perc.lesion_met_largest_component,
              und_metrics.global_efficiency,
              und_metrics.density]
func_list = zip(func_calls, func_names)

#################
# Do percolation
#################

print 'Building percolation data...'
print 'Graphs: ' + str(graph_names)
#if not save_files and repeats > 10:
#    '\nAre you sure you don\'t want to save results?\n'
# Matrices for targeted attack
targ = np.zeros((len(graph_names), len(func_list), len(lesion_list), repeats))

# Percolation
for ri in np.arange(repeats):
    print 'Lesioning; repeat ' + str(ri + 1) + ' of ' + str(repeats)
    graph_list = construct_graph_list_und(graph_names)

    # Cycle over graphs and metric functions
    for gi, G in enumerate(graph_list):
        for fi, (func, func_label) in enumerate(func_list):
            targ[gi, fi, :, ri] = perc.percolate_degree(G.copy(), lesion_list,
                                                        func)
        print ' ... Done ' + str(gi + 1) + '/' + str(len(graph_list))

###############
# Plot results
###############
# Set font type for compatability with adobe if doing editing later
plt.close('all')
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()

col_d = cf.COLORS
graph_col = [col_d['brain'], col_d['configuration'], col_d['small-world'],
             col_d['scale-free'], col_d['pgpa']]
MS = 20
FONTSIZE = 13
FIGSIZE = (4, 3.75)
FACECOLOR = cf.FACE_COLOR

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, facecolor=FACECOLOR)

labels = ['a', 'b']

metric_labels = ['Global efficiency']
x_axis_labels = ['Nodes removed', 'Connection density']

# Plot edge density vs. Global efficiency
for gi, g_name in enumerate(graph_names):
    # get (graph, metric, all lesion proportions, 1st repeat
    dens = np.squeeze(targ[gi, func_names.index('Connection density'), :, 0])
    eff = np.squeeze(targ[gi, func_names.index('Global efficiency'), :, 0])

    ax.scatter(dens, eff, s=MS, label=g_name, color=graph_col[gi])

# Add plotting params
ax.set_xlim([0.09, 0])
ax.set_ylim([0, 0.6])
ax.locator_params(axis='x', n_bins=3)
ax.set_aspect(0.09 / 0.6)

ax.set_xlabel(x_axis_labels[1])
ax.set_ylabel(metric_labels[0])
ax.legend(loc='lower left', prop={'size': FONTSIZE - 3})

fig.set_tight_layout(True)

fig.savefig(op.join(save_dir, 'fig7fs1.png'), dpi=300)
fig.savefig(op.join(save_dir, 'fig7fs1.pdf'), dpi=300)
