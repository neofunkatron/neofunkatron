"""
Fig 6 subplots a and b
Calculate the node betweenness and node-averaged inverse shortest path length
distributions for the brain and the ER and SGPA models (the latter two
averaged over several instantiations).
"""
from __future__ import print_function, division

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from extract import brain_graph
from random_graph.binary_directed import source_growth as sg
from random_graph.binary_directed import target_attraction as ta
from random_graph.binary_directed import random_directed_deg_seq
from metrics import binary_directed as metrics_bd
from network_plot import change_settings

import brain_constants as bc
from config import COLORS

plt.ion()

L = 0.725
BRAIN_SIZE = [7., 7., 7.]
KEYS = ['er', 'rand', 'ta', 'sg', 'sg_topo', 'geom']
PLOT_KEYS = ['rand', 'geom', 'sg']

FACE_COLOR = 'w'
AX_COLOR = 'k'
FONT_SIZE = 20

# parameters for this particular plot
FIG_SIZE = (15, 6)
INSET_COORDINATES = (0.27, 0.6, 0.2, 0.3)
X_LIM_EFFICIENCY = (0, 1.)
Y_LIM_EFFICIENCY = (0, 130)
ALPHA_DEG_VS_CC = 0.7
N_GRAPH_SAMPLES = 100
DEGREE_VS_CLUSTERING_GRAPH_IDX = 0

LW = 3
SCATTER_SIZE = 15
LABELS = ['Random', 'PA', 'SGPA']
BAR_LABELS = ['Rand. ', 'PA ', 'SGPA ']
ERROR_KW = {'ecolor': 'k', 'elinewidth': 2, 'markeredgewidth': 2, 'capsize': 6}

BINS_NODAL_EFFICIENCY = np.linspace(0, 1, 25)
BINCS_NODAL_EFFICIENCY = 0.5 * (BINS_NODAL_EFFICIENCY[:-1] +
                                BINS_NODAL_EFFICIENCY[1:])

SAVE_FILE_NAME = 'model_graphs_with_efficiency.npy'
PLOT_FILE_NAME = 'model_graphs_with_efficiency_plot.npy'
save_fig = True
if save_fig:
    SAVE_DIR = os.environ['DBW_SAVE_CACHE']

# load connectome graph
G_brain, _, _ = brain_graph.binary_directed()


if os.path.isfile(SAVE_FILE_NAME):
    print('File "{}" found.'.format(SAVE_FILE_NAME))
    print('Attempting to open...')
    try:
        graphss = np.load(SAVE_FILE_NAME)[0]
    except Exception, e:
        raise IOError('Error loading data from file "{}"'.format(SAVE_FILE_NAME))
    else:
        print('File loaded successfully!')
else:
    print('looping over {} graph instantiations...'.format(N_GRAPH_SAMPLES))

    graphss = {key: [] for key in KEYS}

    in_degree_brain = G_brain.in_degree().values()
    out_degree_brain = G_brain.out_degree().values()

    for g_ctr in range(N_GRAPH_SAMPLES):

        Gs = {}
        # create directed ER graph
        Gs['er'] = nx.erdos_renyi_graph(
            bc.num_brain_nodes, bc.p_brain_edge_directed, directed=True)

        # create directed degree-controlled random graph
        Gs['rand'] = random_directed_deg_seq(in_sequence=in_degree_brain,
                                             out_sequence=out_degree_brain,
                                             simplify=True)[0]

        # create target attraction graph
        Gs['ta'] = ta(N=bc.num_brain_nodes,
                      N_edges=bc.num_brain_edges_directed, L=L,
                      brain_size=BRAIN_SIZE,)[0]

        # create source growth graph
        Gs['sg'] = sg(N=bc.num_brain_nodes,
                      N_edges=bc.num_brain_edges_directed, L=L,
                      brain_size=BRAIN_SIZE,)[0]

        # create source growth graph with only topo rule
        Gs['sg_topo'] = sg(N=bc.num_brain_nodes,
                           N_edges=bc.num_brain_edges_directed, L=np.inf,
                           brain_size=BRAIN_SIZE)[0]

        # create purely geometric graph
        Gs['geom'] = sg(N=bc.num_brain_nodes,
                        N_edges=bc.num_brain_edges_directed, L=L, gamma=0,
                        brain_size=BRAIN_SIZE)[0]

        for key, G in Gs.items():

            # calculate nodal efficiency
            efficiency_matrix = metrics_bd.efficiency_matrix(G)
            G.nodal_efficiency = np.sum(efficiency_matrix, axis=1) / (len(G.nodes()) - 1)
            G.counts_nodal_efficiency, G.bins_nodal_efficiency = \
                np.histogram(G.nodal_efficiency, bins=BINS_NODAL_EFFICIENCY)

            graphss[key].append(G)

        if (g_ctr + 1) % 1 == 0:
            print('{} of {} samples completed.'.format(g_ctr + 1, N_GRAPH_SAMPLES))

    # Save file so that we don't have to remake everything
    print('Saving file to disk...')

    save_array = np.array([graphss])
    np.save(SAVE_FILE_NAME, save_array)
    print('File "{}" saved successfully'.format(SAVE_FILE_NAME))

print('Taking averages and generating plots...')

# calculate mean and std of nodal efficiency
counts_nodal_efficiency_mean = {}
counts_nodal_efficiency_std = {}

for key, graphs in graphss.items():
    counts_nodal_efficiency_mean[key] = \
        np.array([G.counts_nodal_efficiency for G in graphs]).mean(axis=0)
    counts_nodal_efficiency_std[key] = \
        np.array([G.counts_nodal_efficiency for G in graphs]).std(axis=0)

# calculate mean and std of global efficiencies
for key, graphs in graphss.items():
    global_effs = [G.nodal_efficiency.mean() for G in graphs]
    print('global eff {}: mean = {}, std = {}'.format(
        key, np.mean(global_effs), np.std(global_effs)))

# calculate nodal efficiency for brain
G_brain.efficiency_matrix = metrics_bd.efficiency_matrix(G_brain)
G_brain.nodal_efficiency = np.sum(G_brain.efficiency_matrix, axis=1) /\
    (len(G_brain.nodes()) - 1)

print('global eff brain: {}'.format(np.mean(G_brain.nodal_efficiency)))

# calculate power-law fits for each graph type and brain
power_law_fits = {}
fits_r_squared = {}
for key, graphs in graphss.items():
    gammas = []
    r_squareds = []
    for graph in graphs:
        fit = metrics_bd.power_law_fit_deg_cc(graph)
        gammas.append(fit[0])
        r_squareds.append(fit[2] ** 2)

    power_law_fits[key] = np.array(gammas)
    fits_r_squared[key] = np.array(r_squareds)

power_law_fit_brain = metrics_bd.power_law_fit_deg_cc(G_brain)
power_law_fits['brain'] = power_law_fit_brain[0]
fits_r_squared['brain'] = power_law_fit_brain[2] ** 2

##############################################################################
# plot clustering vs. degree and nodal_efficiencies for brain and three models
##############################################################################
fig, axs = plt.subplots(1, 2, facecolor=FACE_COLOR, figsize=FIG_SIZE,
                        tight_layout=True)

for a_ctr, ax in enumerate(axs):
    # brain
    if a_ctr == 0:
        cc = nx.clustering(G_brain.to_undirected()).values()
        deg = nx.degree(G_brain.to_undirected()).values()
        ax.scatter(deg, cc, s=SCATTER_SIZE, lw=0, alpha=ALPHA_DEG_VS_CC,
                   c=COLORS['brain'], zorder=1000)

    elif a_ctr == 1:
        hist_connectome = ax.hist(G_brain.nodal_efficiency,
                                  bins=BINS_NODAL_EFFICIENCY,
                                  color=COLORS['brain'])

    # other graphs
    if a_ctr == 0:
        # degree vs. clustering example graph
        Gs = [graphss[key][DEGREE_VS_CLUSTERING_GRAPH_IDX]
              for key in PLOT_KEYS]

        for key, G in zip(PLOT_KEYS, Gs):
            cc = nx.clustering(G.to_undirected()).values()
            deg = nx.degree(G.to_undirected()).values()

            ax.scatter(deg, cc, s=SCATTER_SIZE, lw=0, alpha=ALPHA_DEG_VS_CC,
                       c=COLORS[key])

        # Set lims, ticks, and labels
        ax.set_xlim(0, 150)
        ax.set_xticks((0, 50, 100, 150))
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))

        ax.set_xlabel('Degree')
        ax.set_ylabel('Clustering coefficient')

    elif a_ctr == 1:
        # nodal efficiency histogram
        lines_temp = []
        for key in PLOT_KEYS:
            line = ax.plot(BINCS_NODAL_EFFICIENCY,
                           counts_nodal_efficiency_mean[key],
                           color=COLORS[key], lw=LW)
            ax.fill_between(
                BINCS_NODAL_EFFICIENCY,
                counts_nodal_efficiency_mean[key] - counts_nodal_efficiency_std[key],
                counts_nodal_efficiency_mean[key] + counts_nodal_efficiency_std[key],
                color=COLORS[key], alpha=0.5, zorder=100)
            lines_temp.append(line)

            print('{}: nodal eff. peak at = {}, {}'.format(
                key,
                BINCS_NODAL_EFFICIENCY[counts_nodal_efficiency_mean[key].argmax()],
                counts_nodal_efficiency_mean[key].max())
            )

        ax.set_xlim(X_LIM_EFFICIENCY)
        ax.set_ylim(Y_LIM_EFFICIENCY)

        ax.set_xlabel('Nodal efficiency')
        ax.set_ylabel('Number of nodes')

lines = [h_line[0] for h_line in lines_temp] + [hist_connectome[-1][0]]
axs[1].legend(lines, LABELS + ['Connectome'], fontsize=FONT_SIZE)

texts = ('a', 'b')
for ax, text in zip(axs, texts):
    change_settings.set_all_colors(ax, AX_COLOR)
    change_settings.set_all_text_fontsizes(ax, FONT_SIZE)
    ax.text(0.05, 0.95, text, fontsize=20, fontweight='bold',
            transform=ax.transAxes,  ha='center', va='center')

# add inset with power-law fit bar plot
ax_inset = fig.add_axes(INSET_COORDINATES)
gamma_means = [power_law_fits[key].mean() for key in PLOT_KEYS + ['brain']]
gamma_stds = [power_law_fits[key].std() for key in PLOT_KEYS + ['brain']]
gamma_median_r_squareds = [np.median(fits_r_squared[key])
                           for key in PLOT_KEYS + ['brain']]
colors = [COLORS[key] for key in PLOT_KEYS]

bar_width = .8
x_pos = np.arange(len(PLOT_KEYS)) - bar_width/2

ax_inset.bar(x_pos, gamma_means[:-1], width=bar_width, color=colors,
             yerr=gamma_stds[:-1], error_kw=ERROR_KW)
ax_inset.bar([-bar_width/2 + len(PLOT_KEYS)], power_law_fits['brain'],
             width=bar_width, color=COLORS['brain'])
ax_inset.set_xticks(np.arange(len(PLOT_KEYS) + 1))
ax_inset.set_xticklabels(BAR_LABELS + ['Connectome'], rotation='vertical')

ax_inset.set_yticks([0, -0.2, -0.4, -0.6, -.8, -1.])
ax_inset.set_ylabel(r'$\gamma$')

change_settings.set_all_colors(ax_inset, AX_COLOR)
change_settings.set_all_text_fontsizes(ax_inset, FONT_SIZE)

print('gamma means')
print(zip(PLOT_KEYS + ['brain'], gamma_means))
print('gamma stds')
print(zip(PLOT_KEYS + ['brain'], gamma_stds))
print('median R^2s')
print(zip(PLOT_KEYS + ['brain'], gamma_median_r_squareds))

if save_fig:
    fig.savefig('fig6_ab.pdf', dpi=300)
    fig.savefig('fig6_ab.png', dpi=300)
