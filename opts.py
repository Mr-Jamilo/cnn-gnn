#reference: https://github.com/Delphboy/SuperCap/blob/main/opts.py

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Set a fixed seed. Set as -1 for no seed.')
    parser.add_argument('--batch_size', type=int, default=64, help='Set training and testing batch_size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--size', type=str, default='medium', help='Size of the model. tiny|small|medium|big')
    parser.add_argument('--graph_layer_type', type=str, default='GCN', help='Type of layer during graph construction. GCN|GAT|GIN')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--k_neighbours', type=int, default=7, help='Number of edges for each node')
    parser.add_argument('--stochastic_path', type=float, default=0.1, help='Rate of paths to drop')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--dataset_directory', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=150, help='Epochs')
    args = parser.parse_args()

    assert args.size == 'tiny' or args.size == 'small' or args.size == 'medium' or args.size == 'big', 'Size should be tiny|small|medium|big'
    assert args.graph_layer_type == 'GCN' or args.graph_layer_type == 'GAT' or args.graph_layer_type == 'GIN', 'Graph layer type should be GCN|GAT|GIN'
    assert args.epochs > 0, 'Epochs should be greater than 0'
    assert args.threshold >= 0 and args.threshold < 1, 'Threshold should be between 0 and 1'
    assert args.k_neighbours > 0, 'K_neighbours should be greater than 0'
    assert args.weight_decay > 0, 'Weight decay should be greater than 0'
    assert args.batch_size > 0, 'Batch size should be greater than 0'

    return args

