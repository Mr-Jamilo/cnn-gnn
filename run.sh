#!/bin/bash

source .venv/bin/activate

sizes=("tiny" "small" "medium" "big")
knn_arr=(3 5 7 9)
sto_path=0

for size in "${sizes[@]}"; do
  if [ $size = "big" ]; then
    sto_path=0.3
  else
    sto_path=0.1
  fi
  for knn in "${knn_arr[@]}"; do
    python3 gnn.py --size $size \
      --dataset_directory "./dataset" \
      --learning_rate 1e-4 \
      --weight_decay 1e-3 \
      --threshold 0.5 \
      --graph_layer_type "GAT" \
      --epochs 150 \
      --stochastic_path $sto_path \
      --k_neighbours $knn \
      --batch_size 16 \
      --seed 42
  done
done
