import random
import unittest

import numpy as np
import torch
from networkx.algorithms.isomorphism import is_isomorphic
from torch_geometric.loader import DataLoader

from complex_edit_operations.yakindu import get_complex_add_transition_edit_operation, \
    get_complex_add_region_with_entry_operation
from m2_generator.edit_operation.edit_operation import edge_match
from m2_generator.edit_operation.pallete import Pallete, add_inv_edges
from m2_generator.model2graph.model2graph import get_graph_from_model
from m2_generator.neural_model.data_generation import sequence2data
from m2_generator.neural_model.generative_model import GenerativeModel, sample_graph
from tests.test_edit_operation import path_model_yakindu, path_metamodel_yakindu, plot_graph

seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def node_match(n1, n2):
    return n1['type'] == n2['type']


class TestNeuralModel(unittest.TestCase):
    def test_training(self):
        model = get_graph_from_model(path_model_yakindu, [path_metamodel_yakindu])
        pallete = Pallete(path_metamodel_yakindu, 'Statechart')

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
            print(f'Epoch {e} loss {round(total_loss / len(loader), 4)}')

        model.eval()
        samples = [sample_graph(pallete.initial_graphs[0], pallete, model, 50, debug=False) for i in range(5)]
        for j, s in enumerate(samples):
            print(len(s), len(G_new))
            print(len(s.edges), len(G_new.edges))
            print(is_isomorphic(s, G_new, node_match, edge_match))
            plot_graph(f'sample {j}', s)

        samples = [sample_graph(pallete.initial_graphs[0], pallete, model, 50, debug=False) for i in range(100)]
        iso = [s for s in samples if is_isomorphic(s, G_new, node_match, edge_match)]
        print(f'Proportion isomorphic {len(iso) / len(samples)}')


if __name__ == '__main__':
    unittest.main()
