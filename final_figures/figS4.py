
import numpy as np
import matplotlib.pyplot as plt

from random_graph.binary_directed import biophysical_indegree, biophysical_reverse_outdegree
from extract.brain_graph import binary_directed as brain_graph
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

import color_scheme
import in_out_plot_config as cf


fig,axs = plt.subplots(1,facecolor='w',figsize=(8,6))
fig.subplots_adjust(bottom=0.15)

#G, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
#                                              N_edges=bc.num_brain_edges_directed,
#                                              L=.75)

G,_,_ = brain_graph()

outdeg = G.out_degree()
indeg = G.in_degree()
nodes = G.nodes()
sumdeg = [float(outdeg[node]+indeg[node]) for node in nodes]

prop_indeg = [indeg[node]/sumdeg[node] for node in nodes]
bins = np.linspace(0,1,11)
axs.hist(prop_indeg,bins,facecolor=[0.8,0.1,0.1])
axs.set_xlabel('Proportion in-degree',fontsize=26)
axs.set_ylabel('Count',fontsize=26)
xticks = [0,0.25,0.5,0.75,1.0]
yticks = np.arange(0,100,20)
axs.set_xticks(xticks); axs.set_xticklabels(xticks,size=24)
axs.set_yticks(yticks); axs.set_yticklabels(yticks,size=24)
plt.show(block=False)