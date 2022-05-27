import random
import unittest

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.isomorphism import is_isomorphic
from torch_geometric.data import DataLoader

from m2_generator.edit_operation.edit_operation import edge_match, EditOperation
from m2_generator.edit_operation.pallete import Pallete, add_inv_edges
from m2_generator.model2graph.model2graph import get_graph_from_model
from m2_generator.neural_model.data_generation import sequence2data
from m2_generator.neural_model.generative_model import GenerativeModel, sample_graph
from tests.test_edit_operation import path_model, path_metamodel, plot_graph

seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def node_match(n1, n2):
    return n1['type'] == n2['type']


def get_complex_add_transition_edit_operation():
    pattern_ati = nx.MultiDiGraph()
    pattern_ati.add_node(0, type=['State', 'Choice', 'Exit', 'FinalState',
                                  'Synchronization', 'Entry'], ids={0, 1})
    pattern_ati.add_node(1, type='Transition')
    pattern_ati.add_edge(0, 1, type='outgoingTransitions')
    pattern_ati.add_edge(0, 1, type='incomingTransitions')

    pattern_at = nx.MultiDiGraph()
    pattern_at.add_node(0, type=['State', 'Choice', 'Exit', 'FinalState',
                                 'Synchronization', 'Entry'], ids={0})
    pattern_at.add_node(1, type='Transition')
    pattern_at.add_node(2, type=['State', 'Choice', 'Exit', 'FinalState',
                                 'Synchronization', 'Entry'], ids={1})
    pattern_at.add_edge(0, 1, type='outgoingTransitions')
    pattern_at.add_edge(2, 1, type='incomingTransitions')

    patterns = [pattern_at, pattern_ati]

    return EditOperation(patterns, ids=[0, 1], name='Add Transition Complex')


def get_complex_add_region_with_entry_operation():
    pattern_arwe = nx.MultiDiGraph()
    pattern_arwe.add_node(0, type=['State', 'Statechart'], ids={0})
    pattern_arwe.add_node(1, type='Region')
    pattern_arwe.add_edge(0, 1, type='regions')
    pattern_arwe.add_node(2, type='Entry')
    pattern_arwe.add_edge(1, 2, type='vertices')
    return EditOperation([pattern_arwe], ids=[0], name='Add Region with Entry')


class TestNeuralModel(unittest.TestCase):
    def test_training(self):
        model = get_graph_from_model(path_model, [path_metamodel])
        initial_graph = nx.MultiDiGraph()
        initial_graph.add_node(0, type='Statechart')
        pallete = Pallete(path_metamodel, [initial_graph])

        print('Edit ops before')
        for e in pallete.edit_operations:
            print(e.name)
        pallete.add_complex_edit_operation(get_complex_add_transition_edit_operation())
        pallete.add_complex_edit_operation(get_complex_add_region_with_entry_operation())
        print('Edit ops after')
        for e in pallete.edit_operations:
            print(e.name)

        G_new = pallete.remove_out_of_scope(model)
        epochs = 500
        batch_size = 8
        hidden_dim = 128
        model = GenerativeModel(hidden_dim, pallete.dic_nodes,
                                pallete.dic_edges, pallete.edit_operations)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion_node = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        criterion_action = torch.nn.CrossEntropyLoss(reduction='mean')
        criterion_finish = torch.nn.BCELoss(reduction='mean')

        for e in range(epochs):
            model.train()
            total_loss = 0
            sequence = pallete.graph_to_sequence(G_new)
            sequence = [(add_inv_edges(s[0]), s[1]) for s in sequence]
            listDatas = sequence2data(sequence, pallete)
            loader = DataLoader(listDatas, batch_size=batch_size,
                                num_workers=0,
                                shuffle=False)
            for j, data in enumerate(loader):
                opt.zero_grad()
                action, nodes, finish = model(data.x, data.edge_index,
                                              torch.squeeze(data.edge_attr, dim=1),
                                              data.batch, data.sequence, data.nodes, data.len_seq, data.action)

                nodes = torch.unsqueeze(nodes, dim=2).repeat(1, 1, 2)
                nodes[:, :, 0] = 1 - nodes[:, :, 1]

                L = torch.max(data.len_seq).item()
                gTruth = data.sequence_masked[:, 0:L]
                loss = (criterion_node(nodes.reshape(-1, 2), gTruth.flatten()) +
                        criterion_action(action, data.action) +
                        criterion_finish(finish.flatten(), data.finished.float())) / 3
                total_loss += loss.item()
                loss.backward()
                opt.step()
            print(f'Epoch {e} loss {round(total_loss, 4)}')

        model.eval()
        samples = [sample_graph(initial_graph, pallete, model, 50, debug=False) for i in range(5)]
        for j, s in enumerate(samples):
            print(len(s), len(G_new))
            print(len(s.edges), len(G_new.edges))
            print(is_isomorphic(s, G_new, node_match, edge_match))
            plot_graph(f'sample {j}', s)

        samples = [sample_graph(initial_graph, pallete, model, 50, debug=False) for i in range(100)]
        iso = [s for s in samples if is_isomorphic(s, G_new, node_match, edge_match)]
        print(f'Proportion isomorphic {len(iso)/len(samples)}')


if __name__ == '__main__':
    unittest.main()
