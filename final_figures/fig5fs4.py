from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools
import networkx as nx
import os
import pandas as pd
from sklearn import manifold
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from numpy import concatenate as cc

from extract.brain_graph import binary_directed
import extract.auxiliary as aux
from random_graph.binary_directed import source_growth as sgpa
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

from brain_constants import *
import config

labelsize=11
ticksize=10
legendsize=8


def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

if __name__ == "__main__":
    # this isn't right... the third return object is a distance matrix


    # Get the connectome
    G,_, labels = binary_directed()
    nodes = G.nodes()
    new_labels = {nodes[i]:labels[i] for i in range(len(nodes))}
    distance_matrix=aux.load_brain_dist_matrix(new_labels)

    mds = manifold.MDS(n_components=3, max_iter=1000, eps=1e-10, dissimilarity='precomputed')
    centroids = mds.fit_transform(distance_matrix)

    inter_node_distances = [dist(edge1, edge2)
                            for edge1 in centroids for edge2 in centroids if not all(edge1 == edge2)]

    G_sgpa, _, model_centroids = sgpa()
    model_distances = [model_centroids[edge1][edge2]
                       for edge1 in G_sgpa.nodes()
                       for edge2 in G_sgpa.nodes()
                       if edge1 != edge2]

    fig,axs = plt.subplots(1,facecolor='white',figsize=(4,2.8))
    fig.subplots_adjust(bottom=0.15,left=0.15)

    bins = np.linspace(0,13,51)
    axs.hist(inter_node_distances,bins,facecolor=config.COLORS['brain'],normed=True)
    model_distances_binned,_ = np.histogram(model_distances,bins,normed=True)
    model_bins = bins[0:-1]+(bins[1]-bins[0])/2
    axs.plot(model_bins,model_distances_binned,'-',c='k',lw=3)
    axs.set_xlim([0,13])

    xticks = [0,4,8,12]
    yticks = np.arange(0,0.3,0.05)
    axs.set_ylabel('Probability',fontsize=labelsize)
    axs.set_xlabel('Distance (mm)',fontsize=labelsize)
    axs.set_xticks(xticks); axs.set_yticks(yticks)
    axs.set_xticklabels(xticks,fontsize=ticksize)
    axs.set_yticklabels(yticks,fontsize=ticksize)
    leg=axs.legend(['7mm$^3$ cube','Mouse brain'],prop={'size':legendsize})
    fig.subplots_adjust(bottom=0.2,left=0.2)
    plt.show(block=False)

    fig.savefig('fig5fs4.png', dpi=300)
    fig.savefig('fig5fs4.pdf', dpi=300)


