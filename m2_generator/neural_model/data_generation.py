import networkx as nx
import torch
from torch_geometric.data import Data

from m2_generator.edit_operation.pallete import SPECIAL


def generate_tensors_from_graph(G, pallete, len_seq):
    node_types = []
    for n in range(len(G)):
        node_types.append(pallete.dic_nodes[G.nodes[n]['type']])

    # special nodes, shape: max_len x nodes
    # special_nodes = [[0 for i in range(len(G))]
    #                 for j in range(max_length_special_nodes)]
    # special nodes, shape: nodes x max_len
    special_nodes = [[0 for j in range(pallete.max_len)]
                     for i in range(len(G))]
    special_nodes_masked = [[0 if j < len_seq else -1 for j in range(pallete.max_len)]
                            for i in range(len(G))]
    for n in range(len(G)):
        if 'ids' in G.nodes[n]:
            for idd in G.nodes[n]['ids']:
                special_nodes[n][idd] = 1
                special_nodes_masked[n][idd] = 1
    # edges
    edges = []
    edges_lab = []
    for e in list(G.edges.data()):
        source = e[0]
        target = e[1]
        lab = e[2]['type']
        edges.append([source, target])
        lab = pallete.dic_edges[lab]
        edges_lab.append(lab)
    if len(G) == 1 and edges == []:
        edges.append([0, 0])
        edges_lab.append(pallete.dic_edges[SPECIAL])
    return (torch.tensor(node_types),
            torch.tensor(special_nodes),
            torch.tensor(special_nodes_masked),
            torch.transpose(torch.tensor(edges), 0, 1),
            torch.unsqueeze(torch.tensor(edges_lab), dim=1))


# graphs of the sequence must have ids 0,1,2...len(g)
def graph2data_preaction(G, pallete):
    nT, sN, sNM, edges, edges_lab = generate_tensors_from_graph(G, pallete, 1)
    data = Data(x=nT,
                edge_index=edges,
                edge_attr=edges_lab,
                nodes=torch.tensor(len(G)))
    return data


def graph2data_postaction(G, pallete, len_seq):
    nT, sN, sNM, edges, edges_lab = generate_tensors_from_graph(G, pallete,
                                                                len_seq)
    data = Data(x=nT,
                edge_index=edges,
                edge_attr=edges_lab,
                nodes=torch.tensor(len(G)),
                sequence=sN,
                sequence_masked=sNM)
    return data


# graphs of the sequence must have ids 0,1,2...len(g)
def sequence2data(sequence, pallete):
    result = []
    for j, (G, id_edit) in enumerate(sequence):
        nT, sN, sNM, edges, edges_lab = generate_tensors_from_graph(G,
                                                                    pallete,
                                                                    len(pallete.get_special_nodes(id_edit)))
        if j == 0:
            finished = torch.tensor(1)
        else:
            finished = torch.tensor(0)
        data = Data(x=nT,
                    edge_index=edges,
                    edge_attr=edges_lab,
                    action=torch.tensor(id_edit),
                    nodes=torch.tensor(len(G)),
                    sequence=sN,
                    sequence_masked=sNM,
                    len_seq=torch.tensor(len(pallete.get_special_nodes(id_edit))),
                    finished=finished)
        result.append(data)

    return result


def data2graph(data, pallete):
    G = nx.MultiDiGraph()
    vocab_nodes_2 = {y: x for x, y in pallete.dic_nodes.items()}
    vocab_edges_2 = {y: x for x, y in pallete.dic_edges.items()}
    for n, t in enumerate(data.x.numpy()):
        G.add_node(n, type=vocab_nodes_2[t])
    edges = torch.transpose(data.edge_index, 0, 1)
    for n, e in enumerate(edges.numpy()):
        s = e[0]
        t = e[1]
        n = vocab_edges_2[torch.squeeze(data.edge_attr[n]).item()]
        G.add_edge(s, t, type=n)
    return G
