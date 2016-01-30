"""
Plot fig 3

Connectome degree distribution (with marginals) and proportion in-degree
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import config
import in_out_plot_config as cf
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors
from extract.brain_graph import binary_directed as brain_graph

cf.MARKERSIZE = 25.
cf.FONTSIZE = 12.
ALPHA = 0.5


###################################
# Construct figure object and axes
###################################

fig = plt.figure(figsize=(8, 4.25), facecolor=config.FACE_COLOR)
plt.subplots_adjust(hspace=0.45, wspace=0.45)

left_main_ax = plt.subplot2grid(cf.subplot_divisions, cf.left_main_location,
                                rowspan=cf.left_main_rowspan,
                                colspan=cf.left_main_colspan, aspect='equal')

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

# To make log axes we need to create another axes on top of our existing ones
top_dummy_ax = top_margin_ax.twinx()
right_dummy_ax = right_margin_ax.twiny()

# Extract information for the Allen mouse connectome
G, _, _ = brain_graph()

# Get in- & out-degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg

# Calculate proportion in degree
percent_indeg = indeg / deg.astype(float)
a1 = 1.0

# Left main plot (in vs out degree)
left_main_ax.scatter(indeg, outdeg, c=config.COLORS['brain'], s=cf.MARKERSIZE,
                     lw=0, alpha=ALPHA)

left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')
left_main_ax.set_xlim([0, 125])
left_main_ax.set_ylim([0, 125])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 121, 40))
left_main_ax.set_yticks(np.arange(0, 121, 40))
left_main_ax.legend(loc='best')
left_main_ax.text(150, 150, 'a', fontsize=cf.FONTSIZE + 2, fontweight='bold')

# Top marginal (in-degree)
top_margin_ax.hist(indeg, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                   color=config.COLORS['brain'], normed=True, stacked=True)

# This is for the log-axis for in-degree
indeg_hist = np.histogram(indeg, bins=cf.OUTDEGREE_BINS)
indeg_x = indeg_hist[1][0:len(indeg_hist[0])]
indeg_y = indeg_hist[0]
indeg_y = indeg_y / float(indeg_y.sum())

# Add 1e-10 to the in/out deg to avoid zeros when for plotting logs
top_dummy_ax.plot(indeg_x, indeg_y + 1e-10, lw=2, color='b')
top_dummy_ax.yaxis.tick_right()
top_dummy_ax.yaxis.set_label_position('right')
top_dummy_ax.set_yscale('log')
top_dummy_ax.set_ylim([0.001, 1])

top_margin_ax.set_yticks([0, 0, 5, 1.0])
top_margin_ax.set_ylim([0, 1.0])

# Right marginal (out-degree)
right_margin_ax.hist(outdeg, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                     color=config.COLORS['brain'], orientation='horizontal',
                     normed=True, stacked=True)

# This is for the log-axis for out-degree
outdeg_hist = np.histogram(outdeg, bins=cf.OUTDEGREE_BINS)
outdeg_x = outdeg_hist[1][0:len(outdeg_hist[0])]
outdeg_y = outdeg_hist[0]
outdeg_y = outdeg_y / float(outdeg_y.sum())

right_dummy_ax.plot(outdeg_y + 1e-10, outdeg_x, lw=2, color='b')
right_dummy_ax.xaxis.tick_top()
right_dummy_ax.xaxis.set_label_position('top')
right_dummy_ax.set_xscale('log')
right_dummy_ax.set_xlim([0.001, 1])
top_margin_ax.set_yticks([0, 0.05, 0.1])
top_margin_ax.set_ylim([0, 0.1])


plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),
         visible=False)

right_margin_ax.set_xticks([0, 0.05, 0.1])
right_margin_ax.set_xlim([0, 0.1])

# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg, percent_indeg, s=cf.MARKERSIZE, lw=0,
                      c=config.COLORS['brain'], alpha=ALPHA)

right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_ylim([0., 1.05])
right_main_ax.set_xticks(np.arange(0, 151, 50))
right_main_ax.text(1., 1.2, 'b', fontsize=cf.FONTSIZE+2, fontweight='bold',
                   transform=right_main_ax.transAxes, ha='right')

# Set labels and axis ticks
top_margin_ax.set_ylabel('$P(K_\mathrm{in}=k)$')
right_margin_ax.set_xlabel('$P(K_\mathrm{out}=k)$')

for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax,
                top_dummy_ax, right_dummy_ax]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    temp_ax.tick_params(width=1.)

top_lin_ticks = top_margin_ax.get_yticklabels()
right_lin_ticks = right_margin_ax.get_xticklabels()

top_log_ticks = top_dummy_ax.get_yticklabels()
right_log_ticks = right_dummy_ax.get_xticklabels()

for tick in top_lin_ticks + right_lin_ticks:
    tick.set_color('k')

for tick in top_log_ticks + right_log_ticks:
    tick.set_color('blue')
    tick.set_fontsize(7.5)

for tick in top_log_ticks:
    pos = tick.get_position()
    tick.set_position((0.975, pos[1]))

for tick in right_log_ticks:
    pos = tick.get_position()
    tick.set_position((pos[0], 0.98))

for tick in right_log_ticks + right_lin_ticks:
    tick.set_rotation(270)

fig.subplots_adjust(left=0.125, top=0.925, right=0.95, bottom=0.225)

fig.savefig('fig_3.png', dpi=300)
fig.savefig('fig_3.pdf', dpi=300)