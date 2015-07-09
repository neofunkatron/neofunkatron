"""
Created on Thu Feb 26 12:29:20 2015

@author: wronk

Create figures showing progressive percolation on standard and brain graphs.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#import network_gen as ng
from metrics import percolation as perc
reload(perc)
import config as cf
import os
import os.path as op
import pickle

# Set plots to make
undirected = True
directed = False

###############
# Load graphs
###############
load_dir = os.environ['DBW_SAVE_CACHE']
graph_names_und = ['Mouse', 'Random', 'Small-world', 'Scale-free',
                   'PGPA']
graph_names_dir = ['Mouse', 'Random', 'PGPA']

graph_metrics_und = []
graph_metrics_dir = []

### Metrics arrays are (function/metric x number/prop removed x repeats)

for g_name in graph_names_und:
    # Load undirected graph metrics
    load_fname = op.join(load_dir, g_name + '_undirected_perc.pkl')
    open_file = open(load_fname, 'rb')
    graph_metrics_und.append(pickle.load(open_file))
    open_file.close()

    # Calculate mean and std dev across repeats
    for g_dict in graph_metrics_und:
        g_dict['data_rand_avg'] = np.mean(g_dict['data_rand'], axis=-1)
        g_dict['data_rand_std'] = np.std(g_dict['data_rand'], axis=-1)
        g_dict['data_targ_avg'] = np.mean(g_dict['data_targ'], axis=-1)
        g_dict['data_targ_std'] = np.std(g_dict['data_targ'], axis=-1)

if directed:
    # Load directed graph metrics
    for g_name in graph_names_dir:
        load_fname = op.join(load_dir, g_name + '_directed_perc.pkl')
        open_file = open(load_fname, 'rb')
        graph_metrics_dir.append(pickle.load(open_file))
        open_file.close()

    # Calculate mean and std dev across repeats
    for g_dict in graph_metrics_dir:
        g_dict['data_rand_avg'] = np.mean(g_dict['data_rand'], axis=-1)
        g_dict['data_rand_std'] = np.std(g_dict['data_rand'], axis=-1)
        g_dict['data_targ_avg'] = np.mean(g_dict['data_targ'], axis=-1)
        g_dict['data_targ_std'] = np.std(g_dict['data_targ'], axis=-1)

# Get number of nodes removed and proportion of nodes removed
lesion_list = g_dict['removed_targ']
prop_rm = g_dict['removed_rand']

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
LW = 3.
FONTSIZE = cf.FONT_SIZE
FIGSIZE = (11, 5.5)

FACECOLOR = cf.FACE_COLOR
#LABELCOLOR = cf.LABELCOLOR
#TICKSIZE = cf.TICKSIZE

##########################################################
# Random attack (undirected) with subplot for each metric
##########################################################
if undirected:
    # construct figure
    fig1, ax_list1 = plt.subplots(nrows=1, ncols=len(graph_metrics_und[0]['metrics_list']),
                                  facecolor=FACECOLOR, figsize=FIGSIZE)

    # Loop over each metric and then each graph
    for fi, func_label in enumerate(graph_metrics_und[0]['metrics_label']):
        for gi, g_dict in enumerate(graph_metrics_und):
            # Compute x axis vals, y vals, and std devs
            x = g_dict['removed_rand']
            avg = g_dict['data_rand_avg'][fi, :]
            fill_upper = avg + g_dict['data_rand_std'][fi, :]
            fill_lower = avg - g_dict['data_rand_std'][fi, :]

            # Plot traces
            ax_list1[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                              color=graph_col[gi])
            # Plot std deviation range
            ax_list1[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                      facecolor=graph_col[gi],
                                      interpolate=True, alpha=.3)
        ax_list1[fi].set_title('Random Attack', fontsize=FONTSIZE)
        ax_list1[fi].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
        ax_list1[fi].set_ylabel(func_label, fontsize=FONTSIZE)

    ax_list1[1].legend(loc='best')

    for ax in ax_list1:
        ax.locator_params(axis='both', nbins=8)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(FONTSIZE)
    plt.tight_layout()

    ############################################################
    # Targeted attack (undirected) with subplot for each metric
    ############################################################
    fig2, ax_list2 = plt.subplots(nrows=1, ncols=len(graph_metrics_und[0]['metrics_list']),
                                  facecolor=FACECOLOR, figsize=FIGSIZE)

    for fi, func_label in enumerate(graph_metrics_und[0]['metrics_label']):
        for gi, g_dict in enumerate(graph_metrics_und):
            x = g_dict['removed_targ']
            avg = g_dict['data_targ_avg'][fi, :]
            fill_upper = avg + g_dict['data_targ_std'][fi, :]
            fill_lower = avg - g_dict['data_targ_std'][fi, :]

            ax_list2[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                              color=graph_col[gi])
            ax_list2[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                      facecolor=graph_col[gi],
                                      interpolate=True, alpha=.3)
        ax_list2[fi].set_title('Targeted Attack', fontsize=FONTSIZE)
        ax_list2[fi].set_xlabel('Number of Nodes Removed', fontsize=FONTSIZE)
        ax_list2[fi].set_ylabel(func_label, fontsize=FONTSIZE)

    ax_list2[1].legend(loc='best')

    for ax in ax_list2:
        ax.locator_params(axis='both', nbins=8)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(FONTSIZE)
    ax_list2[1].set_ylim([0, .3])  # Manually set to prevent lower axis < 0
    plt.tight_layout()


########################################################
# Random attack (directed) with subplot for each metric
########################################################
if directed:
    fig3, ax_list3 = plt.subplots(nrows=1, ncols=len(graph_metrics_dir[0]['metrics_list']),
                                  facecolor=FACECOLOR, figsize=FIGSIZE)

    for fi, func_label in enumerate(graph_metrics_dir[0]['metrics_label']):
        for gi, g_dict in enumerate(graph_metrics_dir):
            x = g_dict['removed_rand']
            avg = g_dict['data_rand_avg'][fi, :]
            fill_upper = avg + g_dict['data_rand_std'][fi, :]
            fill_lower = avg - g_dict['data_rand_std'][fi, :]

            ax_list3[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                              color=graph_col[gi])
            ax_list3[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                      facecolor=graph_col[gi],
                                      interpolate=True, alpha=.3)
        ax_list3[fi].set_title('Random Attack', fontsize=FONTSIZE)
        ax_list3[fi].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
        ax_list3[fi].set_ylabel(func_label, fontsize=FONTSIZE)

    ax_list3[1].legend(loc='best')

    for ax in ax_list3:
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(FONTSIZE)
    plt.tight_layout()

    ###########################################################
    # Targeted attack (directed) with subplot for each metric
    ###########################################################
    fig4, ax_list4 = plt.subplots(nrows=1, ncols=len(graph_metrics_dir[0]['metrics_list']),
                                  facecolor=FACECOLOR, figsize=FIGSIZE)

    for fi, func_label in enumerate(graph_metrics_dir[0]['metrics_label']):
        for gi, g_dict in enumerate(graph_metrics_dir):
            x = g_dict['removed_targ']
            avg = g_dict['data_targ_avg'][fi, :]
            fill_upper = avg + g_dict['data_targ_std'][fi, :]
            fill_lower = avg - g_dict['data_targ_std'][fi, :]

            ax_list4[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                              color=graph_col[gi])
            ax_list4[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                      facecolor=graph_col[gi],
                                      interpolate=True, alpha=.3)
        ax_list4[fi].set_title('Targeted Attack', fontsize=FONTSIZE)
        ax_list4[fi].set_xlabel('Number of Nodes Removed', fontsize=FONTSIZE)
        ax_list4[fi].set_ylabel(func_label, fontsize=FONTSIZE)

    ax_list4[1].legend(loc='best')

    for ax in ax_list4:
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(FONTSIZE)
    plt.tight_layout()

#####################
# Combined plot hack
#####################
# Targeted attack (undirected, largest component and efficiency)
# and random (undirected efficiency)

fig5, ax_list5 = plt.subplots(nrows=1, ncols=2, figsize=(10., 5.),
                              facecolor=FACECOLOR)

metrics = graph_metrics_und[0]['metrics_label'][1] * 2 + \
    graph_metrics_und[0]['metrics_label'][0]
labels = ['a', 'b', 'c']

'''
# Random attack, undirected efficiency
for gi, g_dict in enumerate(graph_metrics_und):
    x = g_dict['removed_rand']
    avg = g_dict['data_rand_avg'][1, :]
    fill_upper = avg + g_dict['data_rand_std'][fi, :]
    fill_lower = avg - g_dict['data_rand_std'][fi, :]

    ax_list5[0].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                     color=graph_col[gi])
    ax_list5[0].fill_between(x, fill_upper, fill_lower, lw=0,
                             facecolor=graph_col[gi],
                             interpolate=True, alpha=.4)

ax_list5[0].set_title('Random Attack', fontsize=FONTSIZE)
ax_list5[0].set_xlabel('Prop. Nodes Removed', fontsize=FONTSIZE)
ax_list5[0].set_ylabel(graph_metrics_und[0]['metrics_label'][1],
                       fontsize=FONTSIZE, va='baseline')
                       '''

# Targetted attack, undirected efficiency and largest component
# Roll axis on two data matrices just to reverse order of plot
metric_labels = np.roll(graph_metrics_und[0]['metrics_label'], shift=1, axis=0)
for fi, func_label in enumerate(metric_labels):
    for gi, g_dict in enumerate(graph_metrics_und):

        targ_avg = np.roll(g_dict['data_targ_avg'], shift=1, axis=0)
        targ_std = np.roll(g_dict['data_targ_std'], shift=1, axis=0)

        x = g_dict['removed_targ']
        avg = targ_avg[fi, :]
        fill_upper = avg + targ_std[fi, :]
        fill_lower = avg - targ_std[fi, :]

        ax_list5[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=graph_col[gi])
        ax_list5[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=graph_col[gi],
                                  interpolate=True, alpha=.4)
    ax_list5[fi].set_title('Targeted Attack', fontsize=FONTSIZE)
    ax_list5[fi].set_xlabel('Nodes Removed', fontsize=FONTSIZE)
    ax_list5[fi].set_ylabel(func_label, fontsize=FONTSIZE, va='baseline')

ax_list5[1].legend(loc='best', prop={'size': FONTSIZE - 6})

for ax_i, ax in enumerate(ax_list5):
    ax.locator_params(axis='both', nbins=5)
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)
    ax.text(0.06, .92, labels[ax_i], color='k', fontsize=FONTSIZE,
            fontweight='bold', transform=ax.transAxes)
ax_list5[0].set_xlim([0, 426])
ax_list5[1].set_xlim([0, 426])
ax_list5[1].set_ylim([0, 450])
ax_list5[0].set_ylim([0, .3])  # Manually set to prevent lower axis < 0
plt.tight_layout(w_pad=2)
plt.draw()

#fig5.savefig('/home/wronk/Builds/lesion_fig_poster.png', transparent=True)
