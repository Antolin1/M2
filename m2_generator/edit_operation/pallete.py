import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic, DiGraphMatcher

from m2_generator.edit_operation.edit_operation import edge_match, EditOperation
from m2_generator.edit_operation.edit_operation_generation import get_edit_operations

SEP_INV = '_inv'
SPECIAL = '<special>'


def node_match(n1, n2):
    return n1['type'] == n2['type']


def node_match_list(n1, n2):
    ty1 = n1['type']
    ty2 = n2['type']
    if isinstance(ty1, list) and isinstance(ty2, list):
        return len([t for t in ty1 if t in ty2]) > 0
    elif isinstance(ty1, list):
        return ty2 in ty1
    elif isinstance(ty2, list):
        return ty1 in ty2
    else:
        return ty1 == ty2


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
    edges += [SPECIAL]
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


def get_initial_graph(root_element):
    initial_graph = nx.MultiDiGraph()
    initial_graph.add_node(0, type=root_element)
    return initial_graph


class Pallete:
    # editOperations: {x:y} x is id and y is object edit op
    # dic_nodes: {x:y} x is str and y is id (same to dic_edges)

    def __init__(self, path_metamodel, root_element):
        self.root_element = root_element
        self.path_metamodel = path_metamodel
        self.atomic_edit_operations = get_edit_operations(path_metamodel)
        self.initial_graphs = [get_initial_graph(root_element)]
        self.complex_edit_operations = []
        self.dic_nodes = compute_dic_nodes(self.atomic_edit_operations, self.initial_graphs)
        self.dic_edges = compute_dic_edges(self.atomic_edit_operations, self.initial_graphs)
        self.edit_operations = self.complex_edit_operations + self.atomic_edit_operations
        self.max_len = max([len(e.ids) for e in self.edit_operations])

    def graph_to_sequence(self, G):
        list_ids = list(range(0, len(self.edit_operations)))
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

    def remove_out_of_scope(self, G, pattern=False):
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

        if not pattern:
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

    def remove_out_of_scope_edit_operations(self, edit_operation):
        patterns_new = [self.remove_out_of_scope(nx.MultiDiGraph(g), pattern=True) for g in edit_operation.patterns]
        return EditOperation(patterns_new, edit_operation.ids, edit_operation.name)

    def reorder_atomic_edit_operations(self, edit_operation):
        used = []
        for p1 in edit_operation.patterns:
            for e in self.atomic_edit_operations:
                for p in e.patterns:
                    GM = DiGraphMatcher(p1, p, node_match=node_match_list,
                                        edge_match=edge_match)
                    if len(list(GM.subgraph_isomorphisms_iter())) > 0:
                        used.append(e)

        self.atomic_edit_operations = [a for a in self.atomic_edit_operations if a not in used] + \
                                      [a for a in self.atomic_edit_operations if a in used]

    def add_complex_edit_operation(self, edit_operation):
        edit_operation = self.remove_out_of_scope_edit_operations(edit_operation)
        self.complex_edit_operations.append(edit_operation)
        self.reorder_atomic_edit_operations(edit_operation)
        self.edit_operations = self.complex_edit_operations + self.atomic_edit_operations
        self.max_len = max([len(e.ids) for e in self.edit_operations])
