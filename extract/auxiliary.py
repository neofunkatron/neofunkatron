"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Functions to extract graphs from Allen mouse connectivity.
"""
from __future__ import print_function, division
import numpy as np
import os
import pandas as pd
import urllib
import aux_random_graphs

DATA_FILE_URL = 'http://www.nature.com/nature/journal/v508/n7495/extref/nature13186-s4.xlsx'
DATA_FILE_NAME = 'nature13186-s4.xlsx'
STRUCTURE_DIRECTORY = '../../mouse_connectivity_data'


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


def load_centroids(labels, data_dir=STRUCTURE_DIRECTORY, in_mm=True):
    """Load centroids."""
    centroids = aux_random_graphs.get_coords()
    centroidsMat = np.zeros([len(labels),3])
    for i,node in enumerate(labels):
        centroidsMat[i,:] = centroids[node]

    if in_mm:
        centroidsMat /= 10.

    return centroidsMat

'''
# This function doesn't work for me because I don't have
# friday_harbor.Ontology...
def mask_specific_structures(structure_list, parent_structures=['CTX'],
                             data_dir=STRUCTURE_DIRECTORY):
    """Return mask for specific structures in super structure.

    Returns boolean mask where True values are structures in structure_list that
    are substructures of parent_structure.

    Args:
        structure_list: list of structure names (with _L or _R appended)
        parent_structure: parent structure
    Returns:
        boolean mask of same length as structure_list"""
    onto = Ontology(data_dir=data_dir)

    # Make sure parent_structures is list
    if not isinstance(parent_structures,list):
        parent_structures = [parent_structures]

    # Get ids of parent structures
    parent_ids = [onto.structure_by_acronym(structure).structure_id \
    for structure in parent_structures]

    # Get ancestors of each structure in structure_list
    ancestors_list = [onto.structure_by_acronym(structure[:-2]).path_to_root \
    for structure in structure_list]

    # Get boolean mask of which structures have ancestors in parent_ids
    mask = [bool(set(parent_ids) & set(ancestors)) \
    for ancestors in ancestors_list]

    return np.array(mask)
'''
