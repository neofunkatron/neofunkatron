"""
Create data for and plot fig 7 supplemental figure 1
Fig7sf1 a: Nodes removed vs. Connection density
Fig7sf1 b: Connection density vs. Global efficiency
"""

import os
import os.path as op
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import config as cf
from config import COLORS
from extract import brain_graph
from metrics import percolation as perc
from metrics import binary_undirected as und_metrics
import brain_constants as bc
from random_graph.binary_directed import source_growth
from random_graph.binary_undirected import random_simple_deg_seq

from config.graph_parameters import LENGTH_SCALE, SW_REWIRE_PROB, BRAIN_SIZE


##############################################################################


# Potentially move this to random_graph/binary_undirected
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
                                       brain_size=BRAIN_SIZE, tries=1000)[0]
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
            G_SGPA_RAND = random_simple_deg_seq(
                sequence=SGPA_degree, brain_size=BRAIN_SIZE, tries=1000)[0]
            graph_list.append(G_SGPA_RAND)

    # Error check that we created correct number of graphs
    if len(graph_list) != len(graphs_to_const):
        raise RuntimeError('Graph list/names don\'t match')

    return graph_list

###################
# Setup figure
###################

# Set font type for compatability with adobe if doing editing later
plt.close('all')
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()

graph_col = [COLORS['brain'], COLORS['random'], COLORS['small-world'],
             COLORS['scale-free'], COLORS['sgpa']]
MS = 20
FONTSIZE = 13
FIGSIZE = (7.5, 3.75)
FACECOLOR = cf.FACE_COLOR
LW = 2.5
labels = ['a', 'b']

##############################################################
fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE,
                            facecolor=FACECOLOR)

metric_labels = ['Global efficiency']
x_axis_labels = ['Nodes removed', 'Connection density']
lesion_type = 'targ'

graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free', 'SGPA']

###############################################################################
# Subplot a: Plot connection density
###############################################################################
edge_di = 2  # Edge density index
max_edges = 426 * 425 / 2.  # Calculate theoretical max edges, but check later

###################
# Load pickled data
###################
graph_metrics_und = []
for g_name in graph_names:
    # Load undirected graph metrics
    load_fname = g_name + '_undirected_perc.pkl'
    open_file = open(load_fname, 'rb')
    graph_metrics_und.append(pickle.load(open_file))
    open_file.close()

    # Calculate mean and std dev across repeats
    for g_dict in graph_metrics_und:
        # Check index of edge density is correct
        assert g_dict['metrics_list'][edge_di] == 'edge_count', (
            'Wrong index for edge density')

        # Check max nodes of edges is correct, and calc connection density
        max_nodes = g_dict['data_targ'][0, 0, :]
        assert np.all(max_nodes == 426), ('Graphs should start with 426 nodes')

        g_dict['density'] = g_dict['data_targ'][edge_di, :, :] / max_edges

####################################################
# Compute connection density averages/stds and plot
####################################################
for gi, g_dict in enumerate(graph_metrics_und):
    x = g_dict['removed_' + lesion_type]

    # Plot mean lines
    avg = np.squeeze(np.mean(g_dict['density'], axis=-1))
    ax_list[0].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                    color=graph_col[gi])

    # Plot std. devs
    std = np.squeeze(np.std(g_dict['density'], axis=-1))
    fill_upper = avg + std
    fill_lower = avg - std

    ax_list[0].fill_between(x, fill_upper, fill_lower, lw=0,
                            facecolor=graph_col[gi], interpolate=True,
                            alpha=.4)


# Plotting properties for the left subplot
ax_list[0].set_xlabel(x_axis_labels[0], fontsize=FONTSIZE)
ax_list[0].set_ylabel('Connection density', fontsize=FONTSIZE)
ax_list[0].set_xlim([0, 426])
ax_list[0].set_ylim([0, 0.0925])

# Difficulty with x locator_params, maybe because of limit?
# ax_list[0].locator_params(axis='x', n_bins=6)
ax_list[0].set_xticks(np.arange(0, 500, 100))
ax_list[0].locator_params(axis='y', n_bins=4)
ax_list[0].set_aspect(426 / 0.0925)
ax_list[0].legend(loc='upper right', prop={'size': FONTSIZE - 1})
ax_list[0].text(0.08, .92, labels[0], color='k', fontsize=FONTSIZE + 1,
                fontweight='bold', transform=ax_list[0].transAxes)

###############################################################################
# Subplot b: Plot connection density vs. efficiency
###############################################################################
# subplot b requires one representative lesioning process

repeats = 1  # Number of times to repeat percolation (should be 1 here)
prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
thresh_list = np.arange(150, -1, -5)
node_order = 426

# Undirected
func_names = ['Largest component', 'Global efficiency', 'Connection density']
func_calls = [perc.lesion_met_largest_component,
              und_metrics.global_efficiency,
              und_metrics.density]
func_list = zip(func_calls, func_names)

#################
# Do percolation
#################
print 'Computing percolation data...'
print 'Graphs: ' + str(graph_names)
targ = np.zeros((len(graph_names), len(func_list), len(lesion_list), repeats))

# Percolation and compute metrics
for ri in np.arange(repeats):
    print 'Lesioning; repeat ' + str(ri + 1) + ' of ' + str(repeats)
    graph_list = construct_graph_list_und(graph_names)

    # Cycle over graphs and metric functions
    for gi, G in enumerate(graph_list):
        for fi, (func, func_label) in enumerate(func_list):
            targ[gi, fi, :, ri] = perc.percolate_degree(G.copy(), lesion_list,
                                                        func)
        print ' ... Done (' + str(gi + 1) + '/' + str(len(graph_list)) + ')'

##########################################################
# Plot subplot b: Connection density vs. Global efficiency
##########################################################

# With calculation complete, now plot conn. dens. vs. global eff.
for gi, g_name in enumerate(graph_names):
    dens = np.squeeze(targ[gi, func_names.index('Connection density'), :, 0])
    eff = np.squeeze(targ[gi, func_names.index('Global efficiency'), :, 0])

    ax_list[1].scatter(dens, eff, s=MS, label=g_name, color=graph_col[gi])

# Set figure plotting params
ax_list[1].set_xlim([0.0925, 0])
ax_list[1].set_ylim([0, 0.6])
ax_list[1].set_aspect(0.0925 / 0.6)
ax_list[1].locator_params(axis='x', n_bins=4)

ax_list[1].set_xlabel(x_axis_labels[1], fontsize=FONTSIZE)
ax_list[1].set_ylabel(metric_labels[0], fontsize=FONTSIZE)
ax_list[1].text(0.08, .92, labels[1], color='k', fontsize=FONTSIZE + 1,
                fontweight='bold', transform=ax_list[1].transAxes)

fig.set_tight_layout(True)

# Save pdf and png versions
fig.savefig('fig7fs1.png', dpi=300)
fig.savefig('fig7fs1.pdf', dpi=300)
