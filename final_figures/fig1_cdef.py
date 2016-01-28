"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp, wronk
clustering_vs_degree_brain_standard_graph.py

Fig 1 cdef, Plot clustering vs. degree for mouse connectome and standard
random graphs.
"""

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx

import config
from config.graph_parameters import SW_REWIRE_PROB
import extract.brain_graph
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOTTING PARAMETERS
FACECOLOR = config.FACE_COLOR
FIGSIZE = (8, 2.5)
FONT_SIZE = 13
MARKER_SIZE = 3
# Colors for brain, random-degree controlled, small-world, scale-free graphs
cols = [config.COLORS['brain'], config.COLORS['random'],
        config.COLORS['small-world'], config.COLORS['scale-free']]
DEG_MAX = 150
DEG_TICKS = [0, 50, 100, 150]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]
graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free']
labels = ['c', 'd', 'e', 'f']  # Upper corner labels for each plot

save_fig = True
if save_fig:
    save_dir = os.environ['DBW_SAVE_CACHE']

plt.ion()
plt.close('all')

########################################################
print 'Building graphs...'

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2.)

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Build standard graphs & get their degree & clustering coefficient
# Configuration model (random with fixed degree sequence)
G_CM = nx.random_degree_sequence_graph(brain_degree, tries=100)
CM_degree = nx.degree(G_CM).values()
CM_clustering = nx.clustering(G_CM).values()

# Watts-Strogatz
G_WS = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)),
                               SW_REWIRE_PROB)
WS_degree = nx.degree(G_WS).values()
WS_clustering = nx.clustering(G_WS).values()

# Barabasi-Albert
G_BA = nx.barabasi_albert_graph(n_nodes, int(round(brain_degree_mean / 2.)))
BA_degree = nx.degree(G_BA).values()
BA_clustering = nx.clustering(G_BA).values()

##################
# Powerlaw fitting
##################

print 'Calculating powerlaw fits...'
deg_clust = zip([brain_degree, CM_degree, WS_degree, BA_degree],
                [brain_clustering, CM_clustering, WS_clustering,
                 BA_clustering])

# Power law fitting here and calculate R^2 vals
reg = []
r_vals = []
for (deg, clust) in deg_clust:
    reg.append(stats.linregress(np.log(deg), np.log(clust)))
    r_vals.append(stats.pearsonr(clust, np.exp(reg[-1][1]) * deg **
                                 reg[-1][0]))

r_squared_vals = [r[0] ** 2 for r in r_vals]

############
# Plot
############
print 'Plotting...'

# Make clustering vs. degree plots
fig, axs = plt.subplots(1, 4, facecolor=FACECOLOR, figsize=FIGSIZE,
                        tight_layout=True)

# Brain
x = np.linspace(0.01, 160, 501)
for ax_i, ax in enumerate(axs):
    deg, clust = deg_clust[ax_i]
    ax.scatter(deg, clust, s=MARKER_SIZE, color=cols[ax_i])
    ax.plot(x, np.exp(reg[ax_i][1]) * x ** reg[ax_i][0], 'k', linestyle='--',
            lw=2)

    # Setting other parameters
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_aspect(DEG_MAX / 1.)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)

    ax.set_title(graph_names[ax_i], fontsize=FONT_SIZE)
    ax.text(.1, .87, labels[ax_i], fontsize=FONT_SIZE-1, fontweight='bold',
            transform=ax.transAxes)
    ax.set_xlabel('Degree')
    set_all_text_fontsizes(ax, FONT_SIZE)
    set_all_colors(ax, 'k')

    # Hide x ticklabels in top row & y ticklabels in right columns
    if ax_i == 0:
        ax.set_ylabel('Clustering\ncoefficient')
        ax.text(0.4, 0.85, r'R$^2$ = %.2f' % r_squared_vals[ax_i],
                fontsize=FONT_SIZE - 2, transform=ax.transAxes)
    else:
        ax.text(0.725, 0.85, '%0.2f' % r_squared_vals[ax_i],
                fontsize=FONT_SIZE - 2, transform=ax.transAxes)
        ax.set_yticklabels('')

fig.set_tight_layout({'pad': 1.02, 'w_pad': 0.4})

if save_fig:
    fig.savefig(os.path.join(save_dir, 'fig1_cdef.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, 'fig1_cdef.pdf'), dpi=300)

plt.show()
