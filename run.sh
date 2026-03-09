#!/bin/bash

source .venv/bin/activate

python3 test.py --size "tiny" \
  --dataset_directory "./dataset" \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --threshold 0.5 \
  --epochs 150 \
  --stochastic_path 0.1 \
  --k_neighbours 9 \
  --batch_size 64 \
  --seed 42 \
