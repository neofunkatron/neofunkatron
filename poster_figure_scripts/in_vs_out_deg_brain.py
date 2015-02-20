"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. outdegree distribution for the Allen Brain mouse connectome.
"""

import numpy as np
import matplotlib.pyplot as plt

from extract.brain_graph import binary_directed as brain_graph
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOT PARAMETERS
FACECOLOR = 'black'
FONTSIZE = 16
NBINS = 15

# load brain graph, adjacency matrix, and labels
G, A, labels = brain_graph()

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

# calculate percent in & percent out degree
percent_indeg = indeg/deg.astype(float)
percent_outdeg = outdeg/deg.astype(float)

# open figure
fig = plt.figure(facecolor=FACECOLOR, tight_layout=True)

ax00 = fig.add_subplot(2,3,1)
ax10 = fig.add_subplot(2,3,4, sharex=ax00)
ax01 = fig.add_subplot(2,3,2, sharey=ax00)
ax11 = fig.add_subplot(2,3,5, sharey=ax10)
ax02 = fig.add_subplot(2,3,3)
ax12 = fig.add_subplot(2,3,6)

# plot out vs. in-degree scatter
ax00.scatter(indeg, outdeg, lw=0)
ax00.set_xlabel('indegree')
ax00.set_ylabel('outdegree')

# plot out & in-degree distributions
ax01.hist(outdeg, bins=NBINS, orientation='horizontal')
ax01.set_ylabel('outdegree')
ax01.set_xlabel('# nodes')
ax01.set_xticks(np.arange(0, 161, 40))

ax10.hist(indeg, bins=NBINS)
ax10.set_xlabel('indegree')
ax10.set_ylabel('# nodes')

# plot percent_indeg & percent_outdeg distributions
ax11.hist(percent_indeg, bins=NBINS)
ax11.set_xlabel('% indegree')
ax11.set_ylabel('# nodes')
ax11.set_xticks(np.arange(0, 1.1, .2))

# plot scatter
ax02.scatter(deg, deg_diff, lw=0)
ax02.set_xlabel('outdegree + indegree')
ax02.set_ylabel('outdegree - indegree')
ax02.set_xticks(np.arange(0, 161, 40))

# plot percent_indeg vs. degree
ax12.scatter(deg, percent_indeg, lw=0)
ax12.set_xlabel('outdegree + indegree')
ax12.set_ylabel('% indegree')
ax12.set_xticks(np.arange(0, 161, 40))
ax12.set_yticks(np.arange(0, 1.1, .2))

for ax in [ax00, ax01, ax10, ax11, ax02, ax12]:
    set_all_text_fontsizes(ax, FONTSIZE)
    set_all_colors(ax, 'white')