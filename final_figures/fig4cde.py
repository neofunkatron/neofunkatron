from __future__ import print_function, division
import matplotlib.pyplot as plt
import metrics.binary_directed as metrics
import networkx as nx
import numpy as np
import os

from extract import brain_graph
from extract.auxiliary import load_brain_dist_matrix
from random_graph.binary_directed import source_growth, target_attraction, random_directed_deg_seq
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import brain_constants as bc
import color_scheme as cs

TEMP_FILE_NAME = 'reciprocity_temp.npy'
RECIPROCITY_FILE_NAME = 'reciprocity.npy'
LS = np.linspace(0, 2, 21)
BRAIN_SIZE = [7., 7., 7.]
N_REPEATS = 100
ALPHA = .3
FONT_SIZE = 20
D_BINS = np.linspace(0, 12, 30)
L = 0.725

if not os.path.isfile(TEMP_FILE_NAME):
    # load brain graph
    print('Loading brain graph...')
    g_brain, a_brain, labels = brain_graph.binary_directed()
    brain_in_deg = g_brain.in_degree().values()
    brain_out_deg = g_brain.out_degree().values()
    # load distance matrix
    d_brain = load_brain_dist_matrix(labels, in_mm=True)

    # make two SG graphs and two TA graphs (each one with either L=0.75 and L=0)
    print('Making example models...')
    g_sg_l_inf, a_sg_l_inf, d_sg_l_inf = source_growth(
        N=bc.num_brain_nodes,
        N_edges=bc.num_brain_edges_directed,
        L=np.inf, gamma=1, brain_size=BRAIN_SIZE,
    )
    g_sg_l_0725, a_sg_l_0725, d_sg_l_0725 = source_growth(
        N=bc.num_brain_nodes,
        N_edges=bc.num_brain_edges_directed,
        L=L, gamma=1, brain_size=BRAIN_SIZE,
    )
    g_ta_l_inf, a_ta_l_inf, d_ta_l_inf = target_attraction(
        N=bc.num_brain_nodes,
        N_edges=bc.num_brain_edges_directed,
        L=np.inf, gamma=1, brain_size=BRAIN_SIZE,
    )
    g_ta_l_0725, a_ta_l_0725, d_ta_l_0725 = target_attraction(
        N=bc.num_brain_nodes,
        N_edges=bc.num_brain_edges_directed,
        L=L, gamma=1, brain_size=BRAIN_SIZE,
    )

    # make graphs and calculate and save reciprocities if this haven't been done already
    if not os.path.isfile(RECIPROCITY_FILE_NAME):
        print('Looping through construction of models for reciprocity calculation...')
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
        print('Loading previous reciprocity calculations...')
        rs = np.load(RECIPROCITY_FILE_NAME)[0]

    print('Crunching numbers...')

    # get reciprocal/nonreciprocal distance histograms for real brain
    r_dists_brain, non_r_dists_brain = metrics.dists_by_reciprocity(a_brain, d_brain)

    # do the same for each of the four graphs
    r_non_r_keys = ['sg_l_075', 'sg_l_inf']
    r_dists = {}
    non_r_dists = {}
    r_dists['sg_l_inf'], non_r_dists['sg_l_inf'] = metrics.dists_by_reciprocity(a_sg_l_inf, d_sg_l_inf)
    r_dists['sg_l_0725'], non_r_dists['sg_l_0725'] = metrics.dists_by_reciprocity(a_sg_l_0725, d_sg_l_0725)
    r_dists['ta_l_inf'], non_r_dists['ta_l_inf'] = metrics.dists_by_reciprocity(a_ta_l_inf, d_ta_l_inf)
    r_dists['ta_l_0725'], non_r_dists['ta_l_0725'] = metrics.dists_by_reciprocity(a_ta_l_0725, d_ta_l_0725)

    r_dists_bincs = {}
    r_dists_cts = {}
    non_r_dists_bincs = {}
    non_r_dists_cts = {}

    for key in r_dists.keys():
        r_cts, r_bins = np.histogram(r_dists[key], bins=D_BINS, normed=True)
        r_bincs = 0.5 * (r_bins[:-1] + r_bins[1:])
        non_r_cts, non_r_bins = np.histogram(non_r_dists[key], bins=D_BINS, normed=True)
        non_r_bincs = 0.5 * (non_r_bins[:-1] + non_r_bins[1:])
        r_dists_cts[key] = r_cts
        r_dists_bincs[key] = r_bincs
        non_r_dists_cts[key] = non_r_cts
        non_r_dists_bincs[key] = non_r_bincs

    # calculate mean and std of reciprocity for each L
    sg_mean = rs['sg'].mean(axis=-1)[1:]
    sg_std = rs['sg'].std(axis=-1)[1:]
    ta_mean = rs['ta'].mean(axis=-1)[1:]
    ta_std = rs['ta'].std(axis=-1)[1:]
    rand_mean = np.ones(rs['LS'].shape, dtype=float)[1:] * rs['rand'].mean(axis=-1)
    rand_std = np.ones(rs['LS'].shape, dtype=float)[1:] * rs['rand'].std(axis=-1)

    brain_recip = metrics.reciprocity(g_brain)

    save_dict = {
        'r_dists_brain': r_dists_brain,
        'non_r_dists_brain': non_r_dists_brain,
        'sg_mean': sg_mean,
        'sg_std': sg_std,
        'ta_mean': ta_mean,
        'ta_std': ta_std,
        'rand_mean': rand_mean,
        'rand_std': rand_std,
        'rs': rs,
        'brain_recip': brain_recip,
        'r_dists': r_dists,
        'non_r_dists': non_r_dists,
        'r_dists_bincs': r_dists_bincs,
        'non_r_dists_bincs': non_r_dists_bincs,
        'r_dists_cts': r_dists_cts,
        'non_r_dists_cts': non_r_dists_cts,
    }
    print('Saving plot data to temp file...')
    np.save(TEMP_FILE_NAME, np.array([save_dict]))
else:
    print('Loading plot data from temp file...')
    all_plot_data = np.load(TEMP_FILE_NAME)[0]
    r_dists_brain = all_plot_data['r_dists_brain']
    non_r_dists_brain = all_plot_data['non_r_dists_brain']
    sg_mean = all_plot_data['sg_mean']
    sg_std = all_plot_data['sg_std']
    ta_mean = all_plot_data['ta_mean']
    ta_std = all_plot_data['ta_std']
    rand_mean = all_plot_data['rand_mean']
    rand_std = all_plot_data['rand_std']
    rs = all_plot_data['rs']
    brain_recip = all_plot_data['brain_recip']
    r_dists = all_plot_data['r_dists']
    non_r_dists = all_plot_data['non_r_dists']
    r_dists_bincs = all_plot_data['r_dists_bincs']
    non_r_dists_bincs = all_plot_data['non_r_dists_bincs']
    r_dists_cts = all_plot_data['r_dists_cts']
    non_r_dists_cts = all_plot_data['non_r_dists_cts']

print('Making plots...')
# make plots
fig, axs = plt.subplots(
    1, 3, facecolor='white', edgecolor='white', figsize=(14, 5), tight_layout=True,
)

# histogram of reciprocal vs nonreciprocal distances
axs[0].set_title('Connectome')
axs[0].set_xlabel('Distance (mm)')
axs[0].set_ylabel('Probability')

axs[0].hist(r_dists_brain.flatten(), bins=D_BINS, lw=0, facecolor='r', normed=True, alpha=0.5)
axs[0].hist(non_r_dists_brain.flatten(), bins=D_BINS, lw=0, facecolor='k', normed=True, alpha=0.5)

axs[0].legend(['Recip.', 'Non-Recip.'], fontsize=FONT_SIZE)

axs[0].set_xticks([0, 4, 8, 12])
axs[0].set_xlim(0, 12)
axs[0].set_ylim(0, .7)

axs[1].set_title('Growth models')
axs[1].set_xlabel('Distance (mm)')

axs[1].hist(r_dists['sg_l_0725'], bins=D_BINS, lw=0, color='r', normed=True, alpha=0.5)
axs[1].hist(non_r_dists['sg_l_0725'], bins=D_BINS, lw=0, color='k', normed=True, alpha=0.5)
axs[1].legend(['SG (L={}) \nRecip.'.format(L), 'SG (L={}) \nNon-Recip.'.format(L)])

axs[1].set_xticks([0, 4, 8, 12])
axs[1].set_yticklabels([])
axs[1].set_xlim(0, 12)
axs[1].set_ylim(0, .7)

# reciprocity vs L for SG, TA, and random (random has no dependence on L)
axs[2].set_title('Varying L')
axs[2].set_xlabel('L (mm)')
axs[2].set_ylabel('Reciprocity coefficient')

axs[2].plot(rs['LS'][1:], sg_mean, color=cs.SRCGROWTH, lw=2)
axs[2].plot(rs['LS'][1:], ta_mean, color=cs.TARGETATTRACTION, lw=2)
axs[2].plot(rs['LS'][1:], rand_mean, color=cs.RANDOM, lw=2)
axs[2].axhline(brain_recip, color=cs.ATLAS, ls='--', lw=2)

axs[2].legend(['SG', 'TA', 'Random', 'Connectome'], fontsize=FONT_SIZE)

axs[2].fill_between(rs['LS'][1:], sg_mean - sg_std, sg_mean + sg_std, color=cs.SRCGROWTH, alpha=ALPHA)
axs[2].fill_between(rs['LS'][1:], ta_mean - ta_std, ta_mean + ta_std, color=cs.TARGETATTRACTION, alpha=ALPHA)
axs[2].fill_between(rs['LS'][1:], rand_mean - rand_std, rand_mean + rand_std, color=cs.RANDOM, alpha=ALPHA)

axs[2].set_xlim(0, 2)
axs[2].set_ylim(0, .7)

axs[0].text(.4, .64, 'c', fontweight='bold', fontsize=FONT_SIZE)
axs[1].text(.4, .64, 'd', fontweight='bold', fontsize=FONT_SIZE)
axs[2].text(.06, .64, 'e', fontweight='bold', fontsize=FONT_SIZE)

for ax in axs:
    set_fontsize(ax, FONT_SIZE)

fig.savefig('fig4cde.pdf')
plt.show(block=True)