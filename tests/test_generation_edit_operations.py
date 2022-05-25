import unittest
import matplotlib.pyplot as plt
import networkx as nx

from m2_generator.edit_operation.edit_operation_generation import get_edit_operations


class TestGeneration(unittest.TestCase):
    def test_generation_yakindu(self):
        path_metamodel = "data/yakindu_simplified.ecore"
        edit_operations = get_edit_operations(path_metamodel)
        print(f'There are {len(edit_operations)} edit operations in Yakindu')
        for edit_operation in edit_operations:
            for i, pattern in enumerate(edit_operation.patterns):
                pattern = nx.DiGraph(pattern)
                plt.figure()
                plt.title(f'{edit_operation.name} pattern {i}')
                pos = nx.spring_layout(pattern)
                nx.draw(pattern, pos, labels=nx.get_node_attributes(pattern, 'type'), with_labels=True)
                nx.draw_networkx_edge_labels(pattern, pos, edge_labels=nx.get_edge_attributes(pattern, 'type'))
                plt.show()


if __name__ == '__main__':
    unittest.main()
