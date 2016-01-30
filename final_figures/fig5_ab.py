"""
Fig 5ab: source growth and target attraction in/out degree and
proportion in degree
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from random_graph.binary_directed import target_attraction, source_growth
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors
import brain_constants as bc
import config
import in_out_plot_config as cf

MARKERSIZE = 25.
FONTSIZE = 12.
ALPHA = 0.5

L = 0.725
BRAIN_SIZE = [7., 7., 7.]

######################################
# Create graphs and calculate metrics
######################################

# create attachment and growth models
G_attachment = target_attraction(N=bc.num_brain_nodes,
                                 N_edges=bc.num_brain_edges_directed, L=L,
                                 gamma=1., brain_size=BRAIN_SIZE)[0]

G_growth = source_growth(N=bc.num_brain_nodes,
                         N_edges=bc.num_brain_edges_directed, L=L, gamma=1.,
                         brain_size=BRAIN_SIZE)[0]

# Get in- & out-degree
indeg_attachment = np.array([G_attachment.in_degree()[node]
                             for node in G_attachment])
outdeg_attachment = np.array([G_attachment.out_degree()[node]
                              for node in G_attachment])
deg_attachment = indeg_attachment + outdeg_attachment

indeg_growth = np.array([G_growth.in_degree()[node] for node in G_growth])
outdeg_growth = np.array([G_growth.out_degree()[node] for node in G_growth])
deg_growth = indeg_growth + outdeg_growth

# Calculate proportion in degree
percent_indeg_attachment = indeg_attachment / deg_attachment.astype(float)
percent_indeg_growth = indeg_growth / deg_growth.astype(float)

# Make plots
fig = plt.figure(figsize=(8, 4.25), facecolor='w')
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

# Left main plot (in vs out degree)
left_main_ax.scatter(indeg_growth, outdeg_growth, c=config.COLORS['sgpa'],
                     s=MARKERSIZE, lw=0, alpha=ALPHA, zorder=3)
left_main_ax.scatter(indeg_attachment, outdeg_attachment,
                     c=config.COLORS['tapa'], s=MARKERSIZE, lw=0, alpha=ALPHA)
left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')
left_main_ax.set_xlim([0, 125])
left_main_ax.set_ylim([0, 125])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 121, 40))
left_main_ax.set_yticks(np.arange(0, 121, 40))
left_main_ax.text(150, 150, 'a', fontsize=FONTSIZE + 2, fontweight='bold')

# Top marginal (in-degree)
top_margin_ax.hist(indeg_attachment, bins=cf.OUTDEGREE_BINS,
                   histtype='stepfilled', color=config.COLORS['tapa'],
                   alpha=ALPHA, label='Target attraction', normed=True,
                   stacked=True)
top_margin_ax.hist(indeg_growth, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                   color=config.COLORS['sgpa'], alpha=ALPHA,
                   label='Source growth', normed=True, stacked=True)

# Right marginal (out-degree)
right_margin_ax.hist(outdeg_attachment, bins=cf.OUTDEGREE_BINS,
                     histtype='stepfilled', color=config.COLORS['tapa'],
                     alpha=ALPHA, orientation='horizontal', normed=True,
                     stacked=True)
right_margin_ax.hist(outdeg_growth, bins=cf.OUTDEGREE_BINS,
                     histtype='stepfilled', color=config.COLORS['sgpa'],
                     alpha=ALPHA, orientation='horizontal', normed=True,
                     stacked=True)
for tick in right_margin_ax.get_xticklabels():
    tick.set_rotation(270)

plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),
         visible=False)

# Set tick marks
top_margin_ax.set_yticks([0, 0.05, 0.1])
top_margin_ax.set_ylim([0, 0.1])
right_margin_ax.set_xticks([0, 0.05, 0.1])
right_margin_ax.set_xlim([0, 0.1025])

top_margin_ax.set_ylabel('$P(k_\mathrm{in})$', va='baseline')
right_margin_ax.set_xlabel('$P(k_\mathrm{out})$', va='top')

# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg_growth, percent_indeg_growth, s=MARKERSIZE, lw=0,
                      c=config.COLORS['sgpa'], alpha=ALPHA, label='SGPA',
                      zorder=3)
right_main_ax.scatter(deg_attachment, percent_indeg_attachment, s=MARKERSIZE,
                      lw=0, c=config.COLORS['tapa'], alpha=ALPHA, label='TAPA')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_xticks(np.arange(0, 151, 50))
right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.text(1., 1.2, 'b', fontsize=FONTSIZE + 2, fontweight='bold',
                   transform=right_main_ax.transAxes, ha='right')
right_main_ax.set_xlim([0., 150.])
right_main_ax.set_ylim([-0.025, 1.025])
right_main_ax.legend(loc=(-0.35, 1.12), prop={'size': 12})

# Set axes properties
for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax]:
    set_all_text_fontsizes(temp_ax, FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    temp_ax.tick_params(width=1.)

fig.subplots_adjust(left=0.125, top=0.925, right=0.95, bottom=0.225)

# Save figure
fig.savefig('fig5_ab.png', dpi=300)
fig.savefig('fig5_ab.pdf', dpi=300)
