"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Functions to extract graphs from Allen mouse connectivity.
"""
from __future__ import print_function, division
from itertools import product as cproduct
import numpy as np
import os
import pandas as pd
import urllib

import aux_random_graphs

DATA_FILE_URL = 'http://www.nature.com/nature/journal/v508/n7495/extref/nature13186-s4.xlsx'
DATA_FILE_NAME = 'nature13186-s4.xlsx'
DIST_DATA_FILE_URL = 'http://www.nature.com/nature/journal/v508/n7495/extref/nature13186-s5.xlsx'
DIST_DATA_FILE_NAME = 'nature13186-s5.xlsx'


def load_W_and_P():
    """Load weight and p-value matrices."""

    if not os.path.exists(DATA_FILE_NAME):
        print('Downloading data to {}...'.format(DATA_FILE_NAME))
        urllib.urlretrieve(DATA_FILE_URL, DATA_FILE_NAME)

    w_ipsi_sheet = pd.read_excel(DATA_FILE_NAME, sheetname='W_ipsi')
    w_ipsi = w_ipsi_sheet.as_matrix()
    w_contra = pd.read_excel(DATA_FILE_NAME, sheetname='W_contra').as_matrix()
    p_ipsi = pd.read_excel(DATA_FILE_NAME, sheetname='PValue_ipsi').as_matrix()
    p_contra = pd.read_excel(DATA_FILE_NAME, sheetname='PValue_contra').as_matrix()

    # Make weight matrix for each side, then concatenate them
    W_L = np.concatenate([w_ipsi, w_contra], 1)
    W_R = np.concatenate([w_contra, w_ipsi], 1)
    W = np.concatenate([W_L, W_R], 0)
    # Make p_value matrix in the same manner
    P_L = np.concatenate([p_ipsi, p_contra], 1)
    P_R = np.concatenate([p_contra, p_ipsi], 1)
    P = np.concatenate([P_L, P_R], 0)

    col_labels = list(w_ipsi_sheet.columns)
    # Add ipsi & contra to col_labels
    col_labels_L = [label.split(' ')[0] + '_L' for label in col_labels]
    col_labels_R = [label.split(' ')[0] + '_R' for label in col_labels]
    labels = col_labels_L + col_labels_R

    return W, P, labels


def load_brain_dist_matrix(labels, in_mm=True):
    """
    Load brain distance matrix.
    :param labels: sequence of labels (i.e., ordering of rows/cols)
    :param in_mm: if True, distance matrix is returned in mm, if False in microns
    :return: distance matrix
    """
    if not os.path.exists(DIST_DATA_FILE_NAME):
        print('Downloading data to {}...'.format(DIST_DATA_FILE_NAME))
        urllib.urlretrieve(DIST_DATA_FILE_URL, DIST_DATA_FILE_NAME)

    data = pd.read_excel(DIST_DATA_FILE_NAME)
    n_nodes = len(labels)
    dists = np.zeros((n_nodes, n_nodes), dtype=float)

    for idx_1, idx_2 in cproduct(range(n_nodes), range(n_nodes)):

        label_1 = labels[idx_1]
        label_2 = labels[idx_2]

        base_1 = label_1[:-2]
        base_2 = label_2[:-2]

        if label_1[-1] == label_2[-1]:
            ext_1 = '_ipsi'
            ext_2 = '_ipsi'
        else:
            ext_1 = '_ipsi'
            ext_2 = '_contra'

        dists[idx_1, idx_2] = data[base_1 + ext_1][base_2 + ext_2]

    if in_mm:
        dists /= 1000

    return dists

"""
def get_coords():
    A = scio.loadmat(STRUCTURE_DATA_FILE_NAME)
    contra_coords = A['cntrltrl_coordnt_dctnry']
    contra_type = contra_coords.dtype
    contra_names = contra_type.names
    contra_dict = {contra_names[k] + '_L': contra_coords[0][0][k][0] for k in range(len(contra_coords[0][0]))}

    ipsi_coords = A['ipsltrl_coordnt_dctnry']
    ipsi_type = ipsi_coords.dtype
    ipsi_names = ipsi_type.names
    ipsi_dict = {ipsi_names[k] + '_R': ipsi_coords[0][0][k][0] for k in range(len(ipsi_coords[0][0]))}

    contra_dict.update(ipsi_dict)

    return contra_dict


def load_centroids(labels, in_mm=True):
    ""Load centroids.""
    centroids = get_coords()

    centroidsMat = np.zeros([len(labels), 3])
    for i,node in enumerate(labels):
        centroidsMat[i, :] = centroids[node]

    if in_mm:
        centroidsMat /= 10.

    return centroidsMat
"""