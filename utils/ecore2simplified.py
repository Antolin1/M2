import glob
import sys
import os
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm

from m2_generator.model2graph.model2graph import get_graph_from_model, serialize_graph_model


def main(args):
    input_files = glob.glob(f'{args.input_path}/*')
    input_graphs = []
    for f in input_files:
        try:
            input_graphs.append(get_graph_from_model(f, []))
        except:
            continue
    for j, sample in tqdm(enumerate(input_graphs), desc='saving models'):
        serialize_graph_model(f'{args.output_path}/{j}.xmi', [args.output_metamodel], 'EPackage', sample)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for generate ecore simplified models')
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--output_metamodel')
    args = parser.parse_args()
    main(args)
