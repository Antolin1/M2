import unittest

import matplotlib.pyplot as plt
import networkx as nx

from m2_generator.model2graph.model2graph import get_graph_from_model


class TestM2G(unittest.TestCase):
    def test_m2g(self):
        path_metamodel = "data/yakindu_simplified.ecore"
        path_model = "data/yakindu_example.xmi"
        G = get_graph_from_model(path_model, [path_metamodel])
        G = nx.DiGraph(G)
        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'type'))
        plt.show()


if __name__ == '__main__':
    unittest.main()
