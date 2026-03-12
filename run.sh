#!/bin/bash

source .venv/bin/activate

python3 gnn.py --size "tiny" \
  --dataset_directory "./dataset" \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --threshold 0.5 \
  --graph_layer_type "GAT" \
  --epochs 1 \
  --stochastic_path 0.1 \
  --k_neighbours 9 \
  --batch_size 32 \
  --seed 42 \
