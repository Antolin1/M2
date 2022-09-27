import networkx as nx
import torch
from torch_geometric.data import Data

from m2_generator.edit_operation.pallete import SPECIAL
from m2_generator.edit_operation.edit_operation import IDS
from m2_generator.model2graph.model2graph import ATTS

# TODO define this var in another place
NUM_ATT = 1
MAX_LEN = 32


def get_att_tensors(G, pallete):
    node_att_types = []
    node_att_val = []
    for n in range(len(G)):
        # att types
        list_atts = list(G.nodes[n][ATTS].keys()) if ATTS in G.nodes[n] else ['<pad>' for i in range(NUM_ATT)]
        list_atts_id = [pallete.vocab_att_type[i] for i in list_atts]
        if len(list_atts_id) < NUM_ATT:
            list_atts_id += [pallete.vocab_att_type['<pad>'] for i in range(NUM_ATT - list_atts_id)]
        node_att_types.append(list_atts_id)

        # att values
        atts_n = []
        if ATTS in G.nodes[n]:
            for att in list(G.nodes[n][ATTS].keys()):
                atts_n.append(pallete.vocab_att_val.to_id_pad(att, MAX_LEN))
            if len(atts_n) < NUM_ATT:
                atts_n += [[pallete.vocab_att_val.vocab['<pad>'] for _ in range(MAX_LEN)]
                           for _ in range(NUM_ATT - len(atts_n))]
        else:
            atts_n += [[pallete.vocab_att_val.vocab['<pad>'] for _ in range(MAX_LEN)]
                       for _ in range(NUM_ATT)]
        node_att_val.append(atts_n)

    return torch.tensor(node_att_types), torch.tensor(node_att_val)  # todo use this function


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
        if IDS in G.nodes[n]:
            for idd in G.nodes[n][IDS]:
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
    att_types, att_val = get_att_tensors(G, pallete)
    data = Data(x=nT,
                edge_index=edges,
                edge_attr=edges_lab,
                nodes=torch.tensor(len(G)),
                att_types=att_types,
                att_val=att_val)
    return data


def graph2data_postaction(G, pallete, len_seq):
    nT, sN, sNM, edges, edges_lab = generate_tensors_from_graph(G, pallete,
                                                                len_seq)
    att_types, att_val = get_att_tensors(G, pallete)
    data = Data(x=nT,
                edge_index=edges,
                edge_attr=edges_lab,
                nodes=torch.tensor(len(G)),
                sequence=sN,
                sequence_masked=sNM,
                att_types=att_types,
                att_val=att_val)
    return data


# graphs of the sequence must have ids 0,1,2...len(g)
def sequence2data(sequence, pallete):
    result = []
    for j, (G, id_edit) in enumerate(sequence):
        nT, sN, sNM, edges, edges_lab = generate_tensors_from_graph(G,
                                                                    pallete,
                                                                    len(pallete.get_special_nodes(id_edit)))
        att_types, att_val = get_att_tensors(G, pallete)
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
                    finished=finished,
                    att_types=att_types,
                    att_val=att_val)
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
