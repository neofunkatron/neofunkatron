from __future__ import print_function, division
import matplotlib.pyplot as plt
import metrics.binary_directed as metrics
import networkx as nx
import numpy as np
import os

from extract import brain_graph
from extract.auxiliary import load_brain_dist_matrix
from random_graph.binary_directed import source_growth, target_attraction, random_directed_deg_seq

import brain_constants as bc

RECIPROCITY_FILE_NAME = 'reciprocity.npy'
LS = np.linspace(0, 2, 21)
BRAIN_SIZE = [7., 7., 7.]
N_REPEATS = 100


# load brain graph
g_brain, a_brain, labels = brain_graph.binary_directed()
brain_in_deg = g_brain.in_degree().values()
brain_out_deg = g_brain.out_degree().values()
# load distance matrix
d_brain = load_brain_dist_matrix(labels, in_mm=True)


# make graphs and calculate and save reciprocities if this haven't been done already
if not os.path.isfile(RECIPROCITY_FILE_NAME):

    algos = {'sg': source_growth, 'ta': target_attraction}

    rs = {'LS': LS}

    for key, algo in algos.items():
        print(key)
        rs[key] = np.nan * np.zeros((len(LS), N_REPEATS), dtype=float)

        for l_ctr, l in enumerate(LS):
            print(l)
            for repeat in range(N_REPEATS):
                print(repeat)
                g, _, _ = algo(
                    bc.num_brain_nodes, bc.num_brain_edges_directed, L=l, gamma=1, brain_size=BRAIN_SIZE
                )
                rs[key][l_ctr, repeat] = metrics.reciprocity(g)

        rs['rand'] = np.nan * np.zeros((N_REPEATS,), dtype=float)

        for repeat in range(N_REPEATS):
            g_rand, _, _ = random_directed_deg_seq(
                in_sequence=brain_in_deg, out_sequence=brain_out_deg,
                simplify=True,
            )
            rs['rand'][repeat] = metrics.reciprocity(g_rand)

    np.save(RECIPROCITY_FILE_NAME, np.array([rs]))
else:
    rs = np.load(RECIPROCITY_FILE_NAME)[0]

# get reciprocal/nonreciprocal distance histograms for real brain
brain_r_mask = (a_brain & a_brain.T).astype(bool)
self_mask = ~np.eye(len(a_brain), dtype=bool)
r_dists_brain = d_brain[brain_r_mask & self_mask].flatten()
nonr_dists_brain = d_brain[a_brain.astype(bool) & (~brain_r_mask) & self_mask].flatten()


# make plots
fig, axs = plt.subplots(
    1, 3, facecolor='white', edgecolor='white', figsize=(7.5, 2.3), dpi=300., tight_layout=True
)

axs[0].hist([r_dists_brain.flatten(), nonr_dists_brain.flatten()], bins=20, lw=0, normed=True)
axs[0].set_xlabel('Distance (mm)')
axs[0].set_ylabel('Probability')
axs[0].set_title('Connectome')