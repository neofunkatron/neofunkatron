from __future__ import division, print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from extract import brain_graph
from random_graph.binary_directed import source_growth
from random_graph.binary_directed import source_growth_total_degree
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

import config
import in_out_plot_config as cf


MARKERSIZE = 25.
FONTSIZE = 12.
ALPHA = 0.5

L = 0.725
GAMMAS = [1, 1.67]
LABELSS = [['a', 'b'], ['c', 'd']]
BRAIN_SIZE = [7., 7., 7.]

for gamma, labels in zip(GAMMAS, LABELSS):
    # create attachment and growth models
    Gbrain, _, _ = brain_graph.binary_directed()

    Gsg_total, _, _ = source_growth_total_degree(
        N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=L,
        gamma=gamma, brain_size=BRAIN_SIZE)

    # Get in- & out-degree
    indeg_brain = np.array([Gbrain.in_degree()[node] for node in Gbrain])
    outdeg_brain = np.array([Gbrain.out_degree()[node] for node in Gbrain])
    deg_brain = indeg_brain + outdeg_brain

    indeg_sg_total = np.array([Gsg_total.in_degree()[node] for node in Gsg_total])
    outdeg_sg_total = np.array([Gsg_total.out_degree()[node] for node in Gsg_total])
    deg_sg_total = indeg_sg_total + outdeg_sg_total


    # Calculate proportion in degree
    percent_indeg_brain = indeg_brain / deg_brain.astype(float)
    percent_indeg_sg_total = indeg_sg_total / deg_sg_total.astype(float)


    # make plots
    fig = plt.figure(figsize=(8, 4.25), facecolor='w')
    plt.subplots_adjust(hspace=0.45, wspace=0.45)

    left_main_ax = plt.subplot2grid(cf.subplot_divisions, cf.left_main_location,
                                    rowspan=cf.left_main_rowspan,
                                    colspan=cf.left_main_colspan,
                                    aspect='equal')

    right_main_ax = plt.subplot2grid(cf.subplot_divisions, cf.right_main_location,
                                     rowspan=cf.right_main_rowspan,
                                     colspan=cf.right_main_colspan)

    top_margin_ax = plt.subplot2grid(cf.subplot_divisions, cf.top_margin_location,
                                     rowspan=cf.top_margin_rowspan,
                                     colspan=cf.top_margin_colspan,
                                     sharex=left_main_ax)

    right_margin_ax = plt.subplot2grid(cf.subplot_divisions,
                                       cf.right_margin_location,
                                       rowspan=cf.right_margin_rowspan,
                                       colspan=cf.right_margin_colspan,
                                       sharey=left_main_ax)

    # Left main plot (in vs out degree)
    left_main_ax.scatter(indeg_brain, outdeg_brain,
                         c=config.COLORS['brain'],
                         s=MARKERSIZE, lw=0, alpha=ALPHA)
    left_main_ax.scatter(indeg_sg_total, outdeg_sg_total, c=config.COLORS['sg'],
                         s=MARKERSIZE, lw=0, alpha=ALPHA, zorder=3)

    left_main_ax.set_xlabel('In-degree')
    left_main_ax.set_ylabel('Out-degree')

    left_main_ax.set_xlim([0, 125])
    left_main_ax.set_ylim([0, 125])
    left_main_ax.set_aspect('auto')
    left_main_ax.set_xticks(np.arange(0, 121, 40))
    left_main_ax.set_yticks(np.arange(0, 121, 40))
    left_main_ax.text(150, 150, labels[0], fontsize=FONTSIZE + 2, fontweight='bold')

    # Top marginal (in-degree)
    top_margin_ax.hist(indeg_brain, bins=cf.OUTDEGREE_BINS,
                       histtype='stepfilled', color=config.COLORS['brain'],
                       alpha=ALPHA, label='Brain', normed=True,
                       stacked=True)
    top_margin_ax.hist(indeg_sg_total, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                       color=config.COLORS['sg'], alpha=ALPHA,
                       label='SG total', normed=True, stacked=True)

    # Right marginal (out-degree)
    right_margin_ax.hist(outdeg_brain, bins=cf.OUTDEGREE_BINS,
                         histtype='stepfilled',
                         color=config.COLORS['brain'], alpha=ALPHA,
                         orientation='horizontal', normed=True, stacked=True)
    right_margin_ax.hist(outdeg_sg_total, bins=cf.OUTDEGREE_BINS,
                         histtype='stepfilled', color=config.COLORS['sg'],
                         alpha=ALPHA, orientation='horizontal', normed=True,
                         stacked=True)
    for tick in right_margin_ax.get_xticklabels():
        tick.set_rotation(270)

    plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),
             visible=False)

    top_margin_ax.set_yticks([0, 0.05, 0.1])
    top_margin_ax.set_ylim([0, 0.1])
    right_margin_ax.set_xticks([0, 0.05, 0.1])
    right_margin_ax.set_xlim([0, 0.1025])

    top_margin_ax.set_ylabel('$P(k_\mathrm{in})$', va='baseline')
    right_margin_ax.set_xlabel('$P(k_\mathrm{out})$', va='top')

    # Right main plot (proportion in vs total degree)
    right_main_ax.scatter(deg_brain, percent_indeg_brain,
                          s=MARKERSIZE, lw=0, c=config.COLORS['brain'],
                          alpha=ALPHA, label='Connectome')
    right_main_ax.scatter(deg_sg_total, percent_indeg_sg_total, s=MARKERSIZE, lw=0,
                          c=config.COLORS['sg'], alpha=ALPHA,
                          label='SGPA(total, gamma = {})'.format(gamma), zorder=3)

    right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    right_main_ax.set_yticks(np.arange(0, 1.1, .25))
    right_main_ax.set_xticks(np.arange(0, 151, 50))
    right_main_ax.set_xlabel('Total degree (in + out)')
    right_main_ax.set_ylabel('Proportion in-degree')
    right_main_ax.text(1., 1.2, labels[1], fontsize=FONTSIZE + 2, fontweight='bold',
                       transform=right_main_ax.transAxes, ha='right')
    right_main_ax.set_xlim([0., 150.])
    right_main_ax.set_ylim([-0.025, 1.025])
    right_main_ax.legend(loc=(-0.35, 1.12), prop={'size': 12})

    for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax]:
        set_all_text_fontsizes(temp_ax, FONTSIZE)
        set_all_colors(temp_ax, cf.LABELCOLOR)
        temp_ax.tick_params(width=1.)

    fig.subplots_adjust(left=0.125, top=0.925, right=0.95, bottom=0.225)

    fig.savefig('fig5sf4{}{}.pdf'.format(*labels), dpi=300)
    fig.savefig('fig5sf4{}{}.png'.format(*labels), dpi=300)


    plt.show(block=False)


    r, p = stats.spearmanr(indeg_sg_total, outdeg_sg_total)
    print('r = {}'.format(r))
    print('p = {}'.format(p))
