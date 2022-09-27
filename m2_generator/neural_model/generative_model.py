import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from m2_generator.edit_operation.edit_operation import IDS
from m2_generator.edit_operation.pallete import add_inv_edges, remove_inv_edges
from m2_generator.neural_model.data_generation import graph2data_preaction, graph2data_postaction


def recommend_action(G_0, pallete, model, debug=False):
    g = pallete.remove_out_of_scope(G_0, pallete)
    g = add_inv_edges(g)
    data = graph2data_preaction(g, pallete)
    batch = torch.tensor([0] * len(g))
    action, h_G, nodeEmbeddings = model.get_actions_h_g_embeddings(data.x, data.edge_index,
                                                                   torch.squeeze(data.edge_attr, dim=1),
                                                                   batch, data.att_types, data.att_val)
    if debug:
        print('Actions: ', F.softmax(action))
    action = torch.topk(action, 1).indices
    if debug:
        print('Action', action.item(), pallete.edit_operations[action.item()].name)

    special_nodes = pallete.get_special_nodes(action.item())
    for j, idd in enumerate(special_nodes):
        if idd == 0:
            data = graph2data_preaction(g, pallete)
            sampled_node = model.get_nodes(h_G, nodeEmbeddings,
                                           batch, None, torch.tensor([action.item()]),
                                           None, data.nodes)
            sampled_node = torch.topk(sampled_node, 1).indices
            g.nodes[sampled_node.item()][IDS] = {idd}
        else:
            data = graph2data_postaction(g, pallete,
                                         j + 1)
            # print('Sequence:',data.sequence)
            sampled_node = model.get_nodes(h_G, nodeEmbeddings,
                                           batch, data.sequence, torch.tensor([action.item()]),
                                           torch.tensor([j + 1]), data.nodes)
            sampled_node = torch.topk(sampled_node, 1).indices
            if IDS not in g.nodes[sampled_node.item()]:
                g.nodes[sampled_node.item()][IDS] = {idd}
            else:
                g.nodes[sampled_node.item()][IDS].add(idd)
    applied = pallete.apply_edit(g, action.item())
    if applied is not None:
        if debug:
            print(f'Action {pallete.edit_operations[action.item()].name} applied')
        return remove_inv_edges(applied)
    else:
        if debug:
            print(f'Cannot apply action {pallete.edit_operations[action.item()].name}')
        return None


def sample_graph(G_0, pallete, model, max_size, debug=False, debug_trials=False, max_trials=100):
    G_aux = nx.MultiDiGraph(G_0)
    finish = False
    step = 0
    trials = 0
    while len(G_aux) < max_size and (not finish):
        G_aux_inv = add_inv_edges(G_aux)
        # sample action
        data = graph2data_preaction(G_aux_inv, pallete)
        batch = torch.tensor([0] * len(G_aux_inv))
        sampled_action, isLast, h_G, nodeEmbeddings = model.get_action_and_finish(
            data.x, data.edge_index,
            torch.squeeze(data.edge_attr, dim=1),
            batch)
        if debug:
            print('Step', step)
            print('Action', sampled_action.item(), pallete.edit_operations[sampled_action.item()].name)
            print('Is last', isLast.item() == 1)

        if isLast.item() == 1:
            finish = True
        sampled_action = sampled_action.item()
        # sample nodes
        special_nodes = pallete.get_special_nodes(sampled_action)
        for j, idd in enumerate(special_nodes):
            if idd == 0:
                data = graph2data_preaction(G_aux_inv, pallete)
                sampled_node = model.get_nodes(h_G, nodeEmbeddings,
                                               batch, None, torch.tensor([sampled_action]),
                                               None, data.nodes)
                G_aux.nodes[sampled_node.item()][IDS] = {idd}
            else:
                G_aux_inv = add_inv_edges(G_aux)
                data = graph2data_postaction(G_aux_inv, pallete,
                                             j + 1)
                # print('Sequence:',data.sequence)
                sampled_node = model.get_nodes(h_G, nodeEmbeddings,
                                               batch, data.sequence, torch.tensor([sampled_action]),
                                               torch.tensor([j + 1]), data.nodes)
                if not IDS in G_aux.nodes[sampled_node.item()]:
                    G_aux.nodes[sampled_node.item()][IDS] = {idd}
                else:
                    G_aux.nodes[sampled_node.item()][IDS].add(idd)
        if debug:
            for n in G_aux:
                if IDS in G_aux.nodes[n]:
                    print('Node type', G_aux.nodes[n]['type'], IDS, G_aux.nodes[n][IDS])

        applied = pallete.apply_edit(G_aux, sampled_action)
        if applied is not None:
            if (trials > 0) and debug_trials:
                print('There have been', trials, 'before')
            trials = 0
            G_aux = applied
            step = step + 1
            if debug:
                print()
        else:
            # print('Cannot apply')
            trials = trials + 1
            for n in G_aux:
                if IDS in G_aux.nodes[n]:
                    del G_aux.nodes[n][IDS]
            finish = False
            if trials == max_trials:
                return G_aux
    return G_aux


class GenerativeModel(nn.Module):

    def __init__(self, hidden_dim, vocab_nodes, vocab_edges, vocab_actions,
                 attention=False, num_layers=2, vocab_att_val=None, vocab_att_type=None):
        super(GenerativeModel, self).__init__()

        self.emb_nodes = nn.Embedding(len(vocab_nodes), hidden_dim)
        self.emb_actions = nn.Embedding(len(vocab_actions), hidden_dim)
        if vocab_att_val and vocab_att_type:
            self.emb_att_value = nn.Embedding(len(vocab_att_val), hidden_dim)
            self.emb_att_type = nn.Embedding(len(vocab_att_type), hidden_dim)

        self.convolution = pyg_nn.Sequential('x, edge_index, edge_type', [
            (pyg_nn.RGCNConv(hidden_dim, hidden_dim,
                             num_relations=len(vocab_edges)), 'x, edge_index, edge_type-> x'),
            nn.ReLU(inplace=True)
        ] * num_layers)

        self.gru = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)

        self.linAction = nn.Linear(hidden_dim, hidden_dim)
        self.linAction_final = nn.Linear(hidden_dim, len(vocab_actions))

        self.linNodes = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linNodes_final = nn.Linear(hidden_dim, 1)
        self.finishedLin = nn.Linear(hidden_dim, hidden_dim)
        self.finishedFinal = nn.Linear(hidden_dim, 1)

        self.globalattention = attention
        if self.globalattention:
            self.globalAttentionFinish = pyg_nn.GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

    def get_actions_h_g_embeddings(self, nodeTypes, edge_index, edge_attr, bs, node_att_type=None, node_att_value=None):
        # node embeddings
        nodeTypes = self.emb_nodes(nodeTypes)
        if node_att_type != None and node_att_value != None:
            # aggregation maybe change
            node_att_value = self.emb_att_value(node_att_value).sum(dim=2)
            node_att_type = self.emb_att_type(node_att_type)
            node_att = node_att_value + node_att_type
            nodeTypes += node_att.sum(dim=1)
        nodeEmbeddings = self.convolution(nodeTypes, edge_index, edge_attr)
        # graph embedding, bxhidden_dim
        h_G = None
        if self.globalattention:
            h_G = self.globalAttentionFinish(nodeEmbeddings, bs)
        else:
            h_G = pyg_nn.global_mean_pool(nodeEmbeddings, bs)
        # infer action
        action = torch.relu(self.linAction(h_G))
        action = self.linAction_final(action)
        return action, h_G, nodeEmbeddings

    # TODO: do it in batch
    # TODO: fix random seed
    def get_action_and_finish(self, nodeTypes, edge_index, edge_attr, bs, node_att_type=None, node_att_value=None):
        action, h_G, nodeEmbeddings = self.get_actions_h_g_embeddings(self, nodeTypes, edge_index, edge_attr, bs,
                                                                      node_att_type, node_att_value)
        m = Categorical(F.softmax(torch.squeeze(action)))
        # infer finished
        final = torch.relu(self.finishedLin(h_G))
        final = torch.sigmoid(self.finishedFinal(final))
        isLast = torch.bernoulli(final)
        return m.sample(), isLast, h_G, nodeEmbeddings

    def get_nodes_sample(self, h_G, nodeEmbeddings,
                         bs, sequence_input, action_input, len_seq, nodes_bs):
        m = Categorical(self.get_nodes(h_G, nodeEmbeddings,
                                       bs, sequence_input, action_input, len_seq, nodes_bs))
        return m.sample()

    def get_nodes(self, h_G, nodeEmbeddings,
                  bs, sequence_input, action_input, len_seq, nodes_bs):

        # action
        sos = self.emb_actions(action_input)
        sos = torch.unsqueeze(sos, dim=1)

        input_gru = None
        if sequence_input == None:
            input_gru = sos
            len_seq = torch.tensor([1])
        else:
            emb_seq = (torch.unsqueeze(sequence_input, 2) *
                       torch.unsqueeze(nodeEmbeddings, dim=1))
            out = scatter(emb_seq, bs, dim=0, reduce="sum")
            input_gru = torch.cat((sos, out), dim=1)

        # h_0 of gru
        h_0 = torch.unsqueeze(h_G, dim=0)
        # pack to gru
        pack = pack_padded_sequence(input_gru, len_seq, batch_first=True,
                                    enforce_sorted=False)

        output, h_n = self.gru(pack, h_0)
        seq_unpacked, lens_unpacked = pad_packed_sequence(output,
                                                          batch_first=True)

        out_gru = torch.repeat_interleave(seq_unpacked, nodes_bs, dim=0)

        nodeEmb_unsq = torch.unsqueeze(nodeEmbeddings,
                                       dim=1).repeat(1, out_gru.shape[1], 1)

        concats = torch.cat((out_gru, nodeEmb_unsq), dim=2)
        concats = torch.relu(self.linNodes(concats))

        # nxlx1
        nodes_final = torch.squeeze(self.linNodes_final(concats), dim=2)
        # n x l
        nodes_final = scatter_softmax(nodes_final, bs, dim=0)
        # if not self.training:
        #    print(nodes_final)
        #    print(nodes_final[:,-1])
        #    print()

        return nodes_final[:, -1]

    def forward(self, nodeTypes, edge_index, edge_attr,
                bs, sequence_input, nodes_bs, len_seq,
                action_input, node_att_type=None, node_att_value=None):

        L = torch.max(len_seq)
        sequence_input = sequence_input[:, 0:L]

        # node embeddings
        N = nodeTypes.shape[0]
        nodeTypes = self.emb_nodes(nodeTypes)

        if node_att_type != None and node_att_value != None:
            # aggregation maybe change
            node_att_value = self.emb_att_value(node_att_value).sum(dim=2)
            node_att_type = self.emb_att_type(node_att_type)
            node_att = node_att_value + node_att_type
            nodeTypes += node_att.sum(dim=1)

        H = nodeTypes.shape[1]
        nodeEmbeddings = self.convolution(nodeTypes, edge_index, edge_attr)
        assert nodeEmbeddings.shape[0] == N
        assert nodeEmbeddings.shape[1] == H

        B = nodes_bs.shape[0]
        # graph embedding, bxhidden_dim
        h_G = None
        if self.globalattention:
            h_G = self.globalAttentionFinish(nodeEmbeddings, bs)
        else:
            h_G = pyg_nn.global_mean_pool(nodeEmbeddings, bs)
        assert h_G.shape[0] == B
        assert h_G.shape[1] == H

        # infer action
        action = torch.relu(self.linAction(h_G))
        action = self.linAction_final(action)

        # infer finished
        final = torch.relu(self.finishedLin(h_G))
        final = torch.sigmoid(self.finishedFinal(final))

        # emulate SOS token, the sos token must be the action
        # action input: bxh
        sos = self.emb_actions(action_input)
        assert sos.shape[0] == B
        assert sos.shape[1] == H
        sos = torch.unsqueeze(sos, dim=1)

        L = sequence_input.shape[1]
        # generate sequence
        emb_seq = (torch.unsqueeze(sequence_input, 2) *
                   torch.unsqueeze(nodeEmbeddings, dim=1))
        assert emb_seq.shape[0] == N
        assert emb_seq.shape[1] == L
        assert emb_seq.shape[2] == H

        # b x max_len x h
        out = scatter(emb_seq, bs, dim=0, reduce="sum")
        assert out.shape[0] == B
        assert out.shape[1] == L
        assert out.shape[2] == H

        input_gru = torch.cat((sos, out), dim=1)
        assert input_gru.shape[0] == B
        assert input_gru.shape[1] == L + 1
        assert input_gru.shape[2] == H

        # h_0 of gru
        h_0 = torch.unsqueeze(h_G, dim=0)
        # pack to gru
        pack = pack_padded_sequence(input_gru, len_seq, batch_first=True,
                                    enforce_sorted=False)

        output, h_n = self.gru(pack, h_0)
        # bxlxh
        seq_unpacked, lens_unpacked = pad_packed_sequence(output,
                                                          batch_first=True)
        assert seq_unpacked.shape[0] == B
        assert seq_unpacked.shape[1] == L
        assert seq_unpacked.shape[2] == H

        # nxlxh
        out_gru = torch.repeat_interleave(seq_unpacked, nodes_bs, dim=0)
        assert out_gru.shape[0] == N
        assert out_gru.shape[1] == L
        assert out_gru.shape[2] == H

        nodeEmb_unsq = torch.unsqueeze(nodeEmbeddings, dim=1).repeat(1,
                                                                     out_gru.shape[1], 1)
        assert nodeEmb_unsq.shape[0] == N
        assert nodeEmb_unsq.shape[1] == L
        assert nodeEmb_unsq.shape[2] == H

        concats = torch.cat((out_gru, nodeEmb_unsq), dim=2)
        assert concats.shape[0] == N
        assert concats.shape[1] == L
        assert concats.shape[2] == (2 * H)

        concats = torch.relu(self.linNodes(concats))
        assert concats.shape[2] == H

        # nxlx1
        nodes_final = torch.squeeze(self.linNodes_final(concats), dim=2)
        # n x l
        nodes_final = scatter_softmax(nodes_final, bs, dim=0)
        assert nodes_final.shape[0] == N
        assert nodes_final.shape[1] == L

        return action, nodes_final, final
