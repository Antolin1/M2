import random

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic

from m2_generator.edit_operation.edit_operation import edge_match

SEP_INV = '_inv'


def node_match(n1, n2):
    return n1['type'] == n2['type']


def compute_dic_nodes(edit_operations, initial_graphs):
    set_nodes = set()
    for edit_operation in edit_operations:
        for pattern in edit_operation.patterns:
            for n in pattern:
                if isinstance(pattern.nodes[n]['type'], list):
                    for t in pattern.nodes[n]['type']:
                        set_nodes.add(t)
                else:
                    set_nodes.add(pattern.nodes[n]['type'])
    for graph in initial_graphs:
        for n in graph:
            if isinstance(graph.nodes[n]['type'], list):
                for t in graph.nodes[n]['type']:
                    set_nodes.add(t)
            else:
                set_nodes.add(graph.nodes[n]['type'])
    nodes = list(set_nodes)
    return {y: x for x, y in enumerate(nodes)}


def compute_dic_edges(edit_operations, initial_graphs):
    set_edges = set()
    for edit_operation in edit_operations:
        for pattern in edit_operation.patterns:
            for _, _, d in pattern.edges(data=True):
                set_edges.add(d['type'])
    for graph in initial_graphs:
        for _, _, d in graph.edges(data=True):
            set_edges.add(d['type'])
    edges = list(set_edges)
    edges += [e + SEP_INV for e in edges]
    return {y: x for x, y in enumerate(edges)}


def add_inv_edges(G):
    G_new = nx.MultiDiGraph(G)
    for e in list(G.edges.data()):
        G_new.add_edge(e[1], e[0], type=e[2]['type'] + SEP_INV)
    return G_new


def remove_inv_edges(G):
    G_new = nx.MultiDiGraph(G)
    remove = []
    for e in G.edges:
        if G[e[0]][e[1]][e[2]]['type'].endswith(SEP_INV):
            remove.append((e[0], e[1], e[2]))
    for s, t, k in remove:
        G_new.remove_edge(s, t, k)
    return G_new


class Pallete:
    # editOperations: {x:y} x is id and y is object edit op
    # dic_nodes: {x:y} x is str and y is id (same to dic_edges)
    def __init__(self, edit_operations,
                 initial_graphs,
                 shuffle=False):
        self.edit_operations = edit_operations
        self.initial_graphs = initial_graphs
        self.dic_nodes = compute_dic_nodes(edit_operations, initial_graphs)
        self.dic_edges = compute_dic_edges(edit_operations, initial_graphs)
        self.shuffle = shuffle
        self.max_len = max([len(e.ids) for e in edit_operations])
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
            re = edit_op.remove_edit(G)
            if re is not None:
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

    def remove_out_of_scope(self, G):
        G_new = nx.MultiDiGraph(G)

        edges_delete = []
        for n in G_new:
            for m in G_new[n]:
                for e in G_new[n][m]:
                    ty = G_new[n][m][e]['type']
                    if ty not in self.dic_edges:
                        edges_delete.append((n, m, e))
        for a, b, e in edges_delete:
            G_new.remove_edge(a, b, e)

        nodes_delete = []
        for n in G_new:
            if G_new.nodes[n]['type'] not in self.dic_nodes:
                nodes_delete.append(n)
        for n in nodes_delete:
            G_new.remove_node(n)

        # relabel
        new_map = {}
        j = 0
        for n in G_new:
            new_map[n] = j
            j = j + 1
        G_new = nx.relabel_nodes(G_new, new_map)
        return G_new
