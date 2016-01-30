"""
Calculate the node betweenness and node-averaged inverse shortest path length distributions for the brain and the ER and SGPA models (the latter two averaged over several instantiations).
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from extract import brain_graph
from network_plot import change_settings

import brain_constants as bc
from config import COLORS
from config.graph_parameters import LENGTH_SCALE

FACE_COLOR = 'w'
AX_COLOR = 'k'
FONT_SIZE = 20

# parameters for this particular plot
FIG_SIZE = (12, 5)
BINS = np.linspace(0, 150, 40)

LOAD_FILE_NAME = 'model_graphs_with_efficiency.npy'


# load brain graph
G_brain, _, _ = brain_graph.binary_directed()

# look for file containing graphs
print('Attempting to open file "{}"...'.format(LOAD_FILE_NAME))
try:
    with open(LOAD_FILE_NAME, 'rb') as f:
        graphs = np.load(LOAD_FILE_NAME)[0]['sg']
except Exception, e:
    raise IOError('Error loading data from file "{}" (please run "fig6.py" first'.format(LOAD_FILE_NAME))
else:
    print('File loaded successfully!')

degrees_sgpa = [nx.degree(g.to_undirected()).values() for g in graphs]

degree_brain = nx.degree(G_brain).values()

ctss = []
bincs = 0.5 * (BINS[:-1] + BINS[1:])

for degrees in degrees_sgpa:
    cts, _ = np.histogram(degrees, bins=BINS, normed=True)
    ctss.append(cts)

ctss = np.array(ctss)
cts_mean = ctss.mean(axis=0)
cts_std = ctss.std(axis=0)

labels = ('a', 'b')
fig, axs = plt.subplots(1, 2, facecolor='white', figsize=FIG_SIZE,
                        tight_layout=True)

axs[0].hist(degree_brain, bins=BINS, color=COLORS['brain'], normed=True)
axs[0].plot(bincs, cts_mean, lw=2, color=COLORS['sgpa'])
axs[0].fill_between(bincs, cts_mean - cts_std, cts_mean + cts_std,
                    color=COLORS['sgpa'], alpha=0.5, zorder=2)
axs[0].set_xlim(0, 150)
axs[0].set_ylim(0, 0.04)
axs[0].set_xticks(np.arange(0, 151, 30))
axs[0].set_yticks(np.linspace(0, 0.04, 5, endpoint=True))
axs[0].set_xlabel('Undirected Degree')
axs[0].set_ylabel('Probability')

brain_handle = axs[1].hist(degree_brain, bins=BINS, color=COLORS['brain'],
                           normed=True)
sgpa_handle = axs[1].plot(bincs, cts_mean, lw=2, color=COLORS['sgpa'])
axs[1].fill_between(bincs, cts_mean - cts_std, cts_mean + cts_std,
                    color=COLORS['sgpa'], alpha=0.5, zorder=2)
axs[1].set_xlim(0, 150)
axs[1].set_ylim(1e-4, 1e-1)
axs[1].set_xticks(np.arange(0, 151, 30))
axs[1].set_xlabel('Undirected Degree')
axs[1].set_ylabel('Log probability')
axs[1].set_yscale('log')
axs[1].legend([brain_handle[-1][0], sgpa_handle[0]], ['Connectome', 'SGPA'])

[change_settings.set_all_text_fontsizes(ax, FONT_SIZE) for ax in axs]
for ax, label in zip(axs, labels):
    ax.text(0.05, 0.95, label, fontsize=20, fontweight='bold',
            transform=ax.transAxes,  ha='center', va='center')

fig.savefig('fig6fs2.png', dpi=300)
fig.savefig('fig6fs2.pdf', dpi=300)
