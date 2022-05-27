import glob
import logging
import random
from argparse import ArgumentParser

import numpy as np
import torch

from m2_generator.edit_operation.pallete import Pallete
from m2_generator.model2graph.model2graph import get_graph_from_model
from m2_generator.neural_model.training import train_generator

logger = logging.getLogger(__name__)


def main(args):
    pallete = Pallete(path_metamodel=args.metamodel, root_element=args.root_object)
    training_files = glob.glob(f'{args.training_dataset}/*')
    training_graphs = [pallete.remove_out_of_scope(get_graph_from_model(f, [args.metamodel])) for f in training_files]
    logger.info(f'Loaded dataset of {len(training_graphs)} elements')
    train_generator(training_graphs, pallete, args)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for training a M2 generator')
    parser.add_argument('--training_dataset', default='./datasets/yakindu_exercise/train',
                        help='Path to the training dataset')
    parser.add_argument('--metamodel', default='./data/yakindu_simplified.ecore')
    parser.add_argument('--root_object', default='Statechart')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    parser.add_argument('--hidden_dim', help='Hidden dimension of the neural model.', type=int, default=64)
    parser.add_argument('--k', help='Montecarlo iterations.', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--model_path', default='models/yakindu_exercise.bin')
    parser.add_argument('--patience', default=10)
    parser.add_argument('--pool', help='Number of processes for the montecarlo decomposition', default=12)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    main(args)
