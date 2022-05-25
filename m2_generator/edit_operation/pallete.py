import random

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic

from m2_generator.edit_operation.edit_operation import edge_match


def node_match(n1, n2):
    return n1['type'] == n2['type']


class Pallete:
    # editOperations: {x:y} x is id and y is object edit op
    # dic_nodes: {x:y} x is str and y is id (same to dic_edges)
    def __init__(self, edit_operations,
                 dic_nodes,
                 dic_edges,
                 initial_graphs,
                 max_len,
                 separator,
                 shuffle=True):
        self.edit_operations = edit_operations
        self.initial_graphs = initial_graphs
        self.dic_nodes = dic_nodes
        self.dic_edges = dic_edges
        self.shuffle = shuffle
        self.max_len = max_len
        self.separator = separator
        # TODO: check consistency

    def graph_to_sequence(self, G):
        list_ids = list(range(0, len(self.edit_operations)))
        if self.shuffle:
            random.shuffle(list_ids)
        else:
            list_ids = sorted(list_ids, reverse=False)

        for intial_graph in self.initial_graphs:
            if is_isomorphic(G, intial_graph,
                             node_match, edge_match):
                return []

        for idd in list_ids:
            edit_op = self.edit_operations[idd]
            re = edit_op.removeEdit(G)
            if re != None:
                re_new = nx.MultiDiGraph(re[0])
                for n in re_new:
                    if 'ids' in re_new.nodes[n]:
                        del re_new.nodes[n]['ids']
                return [(re[0], idd)] + self.graph_to_sequence(re_new)
        return []

    def apply_edit(self, G, idd):
        return self.edit_operations[idd].apply_edit(G)

    def get_special_nodes(self, idd):
        return self.edit_operations[idd].ids
