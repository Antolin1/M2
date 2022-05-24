import unittest
import matplotlib.pyplot as plt
import networkx as nx

from m2_generator.edit_operation.edit_operation_generation import get_edit_operations


class TestGeneration(unittest.TestCase):
    def test_generation_yakindu(self):
        path_metamodel = "data/yakindu_simplified.ecore"
        all_patterns = get_edit_operations(path_metamodel)
        for patterns in all_patterns:
            for pattern in patterns:
                pattern = nx.DiGraph(pattern)
                plt.figure()
                pos = nx.spring_layout(pattern)
                nx.draw(pattern, pos, labels=nx.get_node_attributes(pattern, 'type'), with_labels=True)
                nx.draw_networkx_edge_labels(pattern, pos, edge_labels=nx.get_edge_attributes(pattern, 'type'))
                plt.show()


if __name__ == '__main__':
    unittest.main()
