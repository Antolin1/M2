import unittest

import matplotlib.pyplot as plt
import networkx as nx
from multiset import Multiset
from networkx.algorithms.isomorphism import DiGraphMatcher

from m2_generator.edit_operation.edit_operation_generation import get_edit_operations
from m2_generator.edit_operation.pallete import Pallete
from m2_generator.model2graph.model2graph import get_graph_from_model

path_metamodel = "data/yakindu_simplified.ecore"
path_model = "data/yakindu_example.xmi"


def edge_match(e1, e2):
    t1 = []
    t2 = []
    for e in e1:
        t1.append(e1[e]['type'])
    for e in e2:
        t2.append(e2[e]['type'])
    return Multiset(t2).issubset(Multiset(t1))


def node_match(n1, n2):
    return n1['type'] == n2['type']


def plot_graph(title, graph):
    graph = nx.DiGraph(graph)
    plt.figure()
    plt.title(title)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, labels=nx.get_node_attributes(graph, 'type'), with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'type'))
    plt.show()


def get_problematic_graphs():
    g = nx.MultiDiGraph()
    g.add_node(0, type='Statechart')
    g.add_node(1, type='Region')
    g.add_node(2, type='State')
    g.add_node(3, type='Transition')
    g.add_edge(0, 1, type='regions')
    g.add_edge(1, 2, type='vertices')
    g.add_edge(2, 3, type='incomingTransitions')
    g.add_edge(2, 3, type='outgoingTransitions')

    g1 = nx.MultiDiGraph()
    g1.add_node(0, type='Transition')
    g1.add_node(1, type='State')
    g1.add_edge(1, 0, type='incomingTransitions')
    g1.add_edge(1, 0, type='outgoingTransitions')  # if I remove this, it is no longer isomorphic
    return g, g1


class TestEditOperation(unittest.TestCase):
    def test_generation_yakindu(self):
        edit_operations = get_edit_operations(path_metamodel)
        print(f'There are {len(edit_operations)} edit operations in Yakindu')
        for edit_operation in edit_operations:
            for i, pattern in enumerate(edit_operation.patterns):
                plot_graph(f'{edit_operation.name} pattern {i}', pattern)

    def test_pallete(self):
        model = get_graph_from_model(path_model, [path_metamodel])
        pallete = Pallete(path_metamodel, 'Statechart')

        G_new = pallete.remove_out_of_scope(model)

        print(pallete.dic_edges)
        print(pallete.dic_nodes)
        sequence = pallete.graph_to_sequence(G_new)
        for i, s in enumerate(sequence):
            plot_graph(f'step {i} + {pallete.edit_operations[s[1]].name}', s[0])

    def test_problematic_graph(self):
        pallete = Pallete(path_metamodel, 'Statechart')
        g, g1 = get_problematic_graphs()
        sequence = pallete.graph_to_sequence(g)
        print(len(sequence))
        for i, s in enumerate(sequence):
            plot_graph(f'step {i} + {pallete.edit_operations[s[1]].name}', s[0])

        GM = DiGraphMatcher(g, g1, node_match=node_match,
                            edge_match=edge_match)
        for subgraph in GM.subgraph_isomorphisms_iter():
            print(subgraph)


if __name__ == '__main__':
    unittest.main()
