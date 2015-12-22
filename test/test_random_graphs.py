"""
Tests related to our random graphs.
"""
from __future__ import print_function, division
import networkx as nx
import numpy as np
import unittest

from random_graph import binary_directed as random_graph
from random_graph import binary_undirected as random_graph_bu


class RandomGraphsTestCase(unittest.TestCase):

    def test_all_random_graphs_yield_correct_number_of_nodes_and_edges(self):

        G, A, D = random_graph.target_attraction(N=426, N_edges=2000)
        self.assertEqual(len(G.nodes()), 426)
        self.assertEqual(len(G.edges()), 2000)

        G, A, D = random_graph.source_growth(N=426, N_edges=2000)
        self.assertEqual(len(G.nodes()), 426)
        self.assertEqual(len(G.edges()), 2000)

    def test_undirected_pure_geometric_graph_yields_right_adjacency_matrix(self):

        G, A, D = random_graph_bu.pure_geometric(N=426, N_edges=2000, L=.75, brain_size=[7., 7., 7.])
        self.assertEqual(len(G.nodes()), 426)

        np.testing.assert_array_equal(nx.adjacency_matrix(G).todense().astype(int), A.astype(int))


if __name__ == '__main__':
    unittest.main()