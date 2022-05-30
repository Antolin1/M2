import glob
import json
import logging

import multiprocess as mp
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from networkx.algorithms.isomorphism import is_isomorphic
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import realism.metrics as mt
from constraints.yakindu import inconsistent as inconsistent_yakindu
from constraints.ecore import inconsistent as inconsistent_ecore
from m2_generator.edit_operation.pallete import add_inv_edges, PalleteEncoder, edge_match
from m2_generator.model2graph.model2graph import serialize_graph_model, get_graph_from_model
from m2_generator.neural_model.data_generation import sequence2data
from m2_generator.neural_model.early_stopping import EarlyStopping
from m2_generator.neural_model.generative_model import GenerativeModel, sample_graph
from realism.emd import gaussian_emd, compute_mmd

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
    es = EarlyStopping(opt, model, f'{args.model_path}/pytorch_model.bin', patience=args.patience, mode='min')

    # for g in graphs_train[0:5]:
    #    x, y = pallete.graph_to_sequence(g)[-1]
    #    plot_graph(pallete.edit_operations[y].name, x)

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
    # save pallete
    with open(f'{args.model_path}/pallete.json', 'w') as f:
        json.dump(pallete, f, cls=PalleteEncoder)


def generation(pallete, args):
    with open(f'{args.model_path}/pallete.json') as f:
        pallete.from_json(json.load(f))
    checkpoint = torch.load(f'{args.model_path}/pytorch_model.bin', map_location=torch.device('cpu'))
    model = GenerativeModel(args.hidden_dim, pallete.dic_nodes,
                            pallete.dic_edges, pallete.edit_operations)
    model.load_state_dict(checkpoint['model_state_dict'])
    samples = [sample_graph(pallete.initial_graphs[0], pallete, model, args.max_size, debug=False)
               for _ in tqdm(range(args.n_samples), desc='Generation process')]
    for j, sample in tqdm(enumerate(samples), desc='saving models'):
        serialize_graph_model(f'{args.output_path}/{j}.xmi', [args.metamodel], args.root_object, sample)


def node_match(n1, n2):
    return n1['type'] == n2['type']


def evaluate(pallete, args):
    with open(f'{args.model_path}/pallete.json') as f:
        pallete.from_json(json.load(f))
    if 'yakindu' in args.metamodel:
        inconsistent = inconsistent_yakindu
    elif 'ecore' in args.metamodel:
        inconsistent = inconsistent_ecore
    training_files = glob.glob(f'{args.training_dataset}/*')
    training_graphs = [get_graph_from_model(f, [args.metamodel]) for f in
                       training_files]
    logger.info(f'Loaded a training dataset of {len(training_graphs)} elements')

    test_files = glob.glob(f'{args.test_dataset}/*')
    test_graphs = [get_graph_from_model(f, [args.metamodel]) for f in
                   test_files]
    logger.info(f'Loaded a test dataset of {len(test_graphs)} elements')

    generated_files = glob.glob(f'{args.output_path}/*')
    generated_graphs = [get_graph_from_model(f, [args.metamodel]) for f in
                        generated_files]
    logger.info(f'Loaded a test dataset of {len(generated_graphs)} elements')

    consistent_models = [g for g in generated_graphs if not inconsistent(g)]
    # TODO: check this, inconsistency = 0 ? I don't think so...
    logger.info(f'Proportion of consistent models {len(consistent_models) / len(generated_graphs)}')

    iso = []
    for s in generated_graphs:
        for g in training_graphs:
            if is_isomorphic(s, g, node_match, edge_match):
                iso.append(s)
                break
    iso_prop = len(iso) / len(generated_graphs)
    logger.info(f'Proportion of isomorphic models with respect to the training set: {iso_prop}')

    logger.info('Assessing realism')
    hist_degrees_syn = [nx.degree_histogram(G) for G in consistent_models]
    hist_degrees_real = [nx.degree_histogram(G) for G in test_graphs]
    # MMD degree
    mmd_dist_degree = compute_mmd(hist_degrees_real, hist_degrees_syn, kernel=gaussian_emd)
    logger.info(f'Degree MMD: {mmd_dist_degree}')

    dims = list(pallete.dic_edges.keys())
    hist_mpc_syn = [np.histogram(list(mt.mpc(G, dims).values()), bins=100, range=(0.0, 1.0), density=False)[0]
                    for G in consistent_models]
    hist_mpc_real = [np.histogram(list(mt.mpc(G, dims).values()), bins=100, range=(0.0, 1.0), density=False)[0]
                     for G in test_graphs]
    # MMD MPC
    mmd_dist_mpc = compute_mmd(hist_mpc_real, hist_mpc_syn, kernel=gaussian_emd,
                               sigma=1.0 / 10, distance_scaling=100)
    logger.info(f'Degree MPC: {mmd_dist_mpc}')

    hist_na_syn = [np.histogram(list(mt.node_activity(G, dims)), bins=100, range=(0.0, 1.0), density=False)[0]
                   for G in consistent_models]
    hist_na_real = [np.histogram(list(mt.node_activity(G, dims)), bins=100, range=(0.0, 1.0), density=False)[0]
                    for G in test_graphs]
    # MMD NNA
    mmd_dist_na = compute_mmd(hist_na_real, hist_na_syn, kernel=gaussian_emd,
                              sigma=1.0 / 10, distance_scaling=100)
    logger.info(f'Degree NNA: {mmd_dist_na}')
