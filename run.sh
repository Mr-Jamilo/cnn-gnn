#!/bin/bash
#python3 gnn.py --size "small" --dataset_directory "./dataset" --learning_rate 1e-4 --weight_decay 1e-3 --threshold 0.5 --graph_layer_type "GAT" --epochs 150 --stochastic_path 0.1 --k_neighbours 9 --batch_size 16 --seed 42

source .venv/bin/activate

types=("GCN" "GAT" "GIN")
sizes=("tiny" "small")
knn_arr=(3 5 7 9)
sto_path=0

for type in "${types[@]}"; do
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
        --graph_layer_type $type \
        --epochs 150 \
        --stochastic_path $sto_path \
        --k_neighbours $knn \
        --batch_size 64 \
        --seed 42
    done
  done
done
