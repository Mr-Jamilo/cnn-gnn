# reference: https://github.com/Delphboy/SuperCap/blob/main/opts.py

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Set a fixed seed. Set as -1 for no seed.")
    parser.add_argument("--batch_size", type=int, default=64, help="Set training and testing batch_size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--size", type=str, default="medium", help="Size of the model. tiny|small|medium|big")
    parser.add_argument("--graph_layer_type", type=str, default="GCN", help="Type of layer during graph construction. GCN|GAT|GIN")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay. Set as -1 for no weight decay")
    parser.add_argument("--k_neighbours", type=int, default=7, help="Number of edges for each node")
    parser.add_argument("--stochastic_path", type=float, default=0.1, help="Rate of paths to drop")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--dataset_directory", type=str, default="./dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=150, help="Epochs")

    parser.add_argument("--cnn_extraction_layer", type=str, default="layer4", help="CNN layer to extract features from. layer3|layer4|avgpool")
    parser.add_argument("--cnn_res_blocks", type=str, default="3,4,6,3", help="ResNet block configuration (comma-separated)")

    args = parser.parse_args()

    assert args.size in ["tiny", "small", "medium", "big"], "Size should be tiny|small|medium|big"
    assert args.graph_layer_type in ["GCN", "GAT", "GIN"], "Graph layer type should be GCN|GAT|GIN"
    assert args.epochs > 0, "Epochs should be greater than 0"
    assert args.threshold >= 0 and args.threshold < 1, "Threshold should be between 0 and 1"
    assert args.k_neighbours > 0, "K_neighbours should be greater than 0"
    assert (args.weight_decay < 1 and args.weight_decay > 0) or args.weight_decay == -1, "Weight decay should be between 0 and 1 or equal to -1"
    assert args.batch_size > 0, "Batch size should be greater than 0"
    assert args.cnn_extraction_layer in ["layer3", "layer4", "avgpool"], "CNN extraction layer should be layer3|layer4|avgpool"

    try:
        args.cnn_res_blocks_list = [
            int(x.strip()) for x in args.cnn_res_blocks.split(",")
        ]
        assert len(args.cnn_res_blocks_list) == 4, "CNN res blocks should have 4 values"
    except ValueError:
        raise ValueError('CNN res blocks should be comma-separated integers (e.g., "3,4,6,3")')

    return args
