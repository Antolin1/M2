import unittest

import matplotlib.pyplot as plt
import networkx as nx

from m2_generator.edit_operation.edit_operation_generation import get_edit_operations
from m2_generator.edit_operation.pallete import Pallete
from m2_generator.model2graph.model2graph import get_graph_from_model

path_metamodel = "data/yakindu_simplified.ecore"
path_model = "data/yakindu_example.xmi"


def plot_graph(title, graph):
    graph = nx.DiGraph(graph)
    plt.figure()
    plt.title(title)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, labels=nx.get_node_attributes(graph, 'type'), with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'type'))
    plt.show()


class TestEditOperation(unittest.TestCase):
    def test_generation_yakindu(self):
        edit_operations = get_edit_operations(path_metamodel)
        print(f'There are {len(edit_operations)} edit operations in Yakindu')
        for edit_operation in edit_operations:
            for i, pattern in enumerate(edit_operation.patterns):
                plot_graph(f'{edit_operation.name} pattern {i}', pattern)

    def test_pallete(self):
        model = get_graph_from_model(path_model, [path_metamodel])
        edit_operations = get_edit_operations(path_metamodel)
        initial_graph = nx.MultiDiGraph()
        initial_graph.add_node(0, type='Statechart')
        pallete = Pallete(edit_operations, [initial_graph])

        G_new = pallete.remove_out_of_scope(model)

        print(pallete.dic_edges)
        print(pallete.dic_nodes)
        sequence = pallete.graph_to_sequence(G_new)
        for i, s in enumerate(sequence):
            plot_graph(f'step {i} + {pallete.edit_operations[s[1]].name}', s[0])


if __name__ == '__main__':
    unittest.main()
