import logging

import multiprocess as mp
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm

from m2_generator.edit_operation.pallete import add_inv_edges
from m2_generator.neural_model.data_generation import sequence2data
from m2_generator.neural_model.early_stopping import EarlyStopping
from m2_generator.neural_model.generative_model import GenerativeModel

logger = logging.getLogger(__name__)


def distributed_function(g, pallete):
    sequence = pallete.graph_to_sequence(g)
    sequence = [(add_inv_edges(s[0]), s[1]) for s in sequence]
    return sequence2data(sequence, pallete)


def train_generator(graphs_train, pallete, args):
    criterion_node = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    criterion_action = nn.CrossEntropyLoss(reduction='mean')
    criterion_finish = nn.BCELoss(reduction='mean')

    model = GenerativeModel(args.hidden_dim, pallete.dic_nodes,
                            pallete.dic_edges, pallete.edit_operations).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    es = EarlyStopping(opt, model, args.model_path, patience=args.patience, mode='min')

    logger.info('Starting montecarlo decomposition')
    list_datas_montecarlo = []
    for _ in tqdm(range(args.k), desc='Montecarlo decomposition k'):
        with mp.Pool(args.pool) as pool:
            list_datas = pool.map(lambda x: distributed_function(x, pallete), graphs_train)
            list_datas = [r for rr in list_datas for r in rr]
            list_datas_montecarlo += list_datas
    loader = DataLoader(list_datas_montecarlo, batch_size=args.batch_size, num_workers=3, shuffle=True)
    logger.info('Finished montecarlo decomposition and starting training')

    losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for data in loader:
            opt.zero_grad()
            action, nodes, finish = model(data.x.to(args.device), data.edge_index.to(args.device),
                                          torch.squeeze(data.edge_attr, dim=1).to(args.device),
                                          data.batch.to(args.device), data.sequence.to(args.device),
                                          data.nodes.to(args.device), data.len_seq.to('cpu'),
                                          data.action.to(args.device))
            nodes = torch.unsqueeze(nodes, dim=2).repeat(1, 1, 2)
            nodes[:, :, 0] = 1 - nodes[:, :, 1]
            L = torch.max(data.len_seq).item()
            gTruth = data.sequence_masked[:, 0:L]
            loss = (criterion_node(nodes.reshape(-1, 2), gTruth.flatten().to(args.device)) +
                    criterion_action(action, data.action.to(args.device)) +
                    criterion_finish(finish.flatten(), data.finished.float().to(args.device))) / 3
            total_loss += loss.item()
            loss.backward()
            opt.step()

        avg_loss = round(total_loss / len(loader), 4)
        logger.info(f'Epoch {epoch}, Loss = {avg_loss}')
        losses.append(avg_loss)
        if es.step(avg_loss, epoch):
            break
