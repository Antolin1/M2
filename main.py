import glob
import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from prettytable import PrettyTable

from m2_generator.edit_operation.pallete import Pallete
from m2_generator.model2graph.model2graph import get_graph_from_model
from m2_generator.neural_model.training_generation_evaluation import train_generator, generation, evaluate
from tests.test_neural_model import get_complex_add_transition_edit_operation, \
    get_complex_add_region_with_entry_operation

logger = logging.getLogger(__name__)


def main(args):
    pallete = get_pallete(args)
    if args.train:
        training_files = glob.glob(f'{args.training_dataset}/*')
        training_graphs = [pallete.remove_out_of_scope(get_graph_from_model(f, [args.metamodel])) for f in
                           training_files]
        logger.info(f'Loaded dataset of {len(training_graphs)} elements')
        train_generator(training_graphs, pallete, args)
    elif args.inference:
        generation(pallete, args)
    elif args.evaluate:
        evaluate(pallete, args)
    else:
        raise ValueError('--train, --inference, or --evaluate should be provided.')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_pallete(args):
    pallete = Pallete(path_metamodel=args.metamodel, root_element=args.root_object)
    if args.complex_edit_operations:
        if 'yakindu' in args.metamodel.lower():
            ed1 = get_complex_add_transition_edit_operation()
            ed2 = get_complex_add_region_with_entry_operation()
            pallete.add_complex_edit_operation(ed1)
            pallete.add_complex_edit_operation(ed2)
    return pallete


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for training a M2 generator')
    parser.add_argument('--training_dataset', default='./datasets/yakindu_exercise/train',
                        help='Path to the training dataset')
    parser.add_argument('--test_dataset', default='./datasets/yakindu_exercise/test',
                        help='Path to the test dataset')
    parser.add_argument('--metamodel', default='./data/yakindu_simplified.ecore')
    parser.add_argument('--root_object', default='Statechart')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    parser.add_argument('--hidden_dim', help='Hidden dimension of the neural model.', type=int, default=64)
    parser.add_argument('--k', help='Montecarlo iterations.', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model_path', default='models/yakindu_exercise')
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--pool', help='Number of processes for the montecarlo decomposition', type=int, default=12)
    parser.add_argument('--max_size', help='Maximum size of the generated models', type=int, default=150)
    parser.add_argument('--n_samples', help='Number of samples to generate', type=int, default=500)
    parser.add_argument('--output_path', help='Output folder where the models will be generated',
                        default='generated_models/yakindu_exercise')
    parser.add_argument('--complex_edit_operations', help='If complex edit operations are considered', action='store_true')
    parser.add_argument('--train', help='If training phase', action='store_true')
    parser.add_argument('--inference', help='If inference phase', action='store_true')
    parser.add_argument('--evaluate', help='Evaluate of generated models', action='store_true')
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

    file = logging.FileHandler(os.path.join(args.model_path, 'info.log'))
    file.setLevel(level=logging.INFO)
    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info('Configuration:\n{}'.format(config_table))

    main(args)
