#!/bin/bash

source .venv/bin/activate

run_cnn_binary() {
    local m_res_blocks=$1

    python3 cnn-binary.py \
        --dataset_directory "./dataset" \
        --learning_rate 1e-4 \
        --weight_decay 1e-3 \
        --threshold 0.5 \
        --epochs 150 \
        --batch_size 32 \
        --seed 42 \
        --cnn_res_blocks "$m_res_blocks"
}

run_gnn() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4

    python3 gnn-multilabel.py --size "$m_size" \
        --dataset_directory "./dataset" \
        --learning_rate 1e-5 \
        --weight_decay 1e-3 \
        --threshold 0.6 \
        --graph_layer_type "$m_type" \
        --epochs 150 \
        --stochastic_path "$m_sto" \
        --k_neighbours "$m_knn" \
        --batch_size 32 \
        --seed 42
} 

run_cnn_gnn() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4
    local m_extract=$5
    local m_res_blocks=$6

    python3 cnn-gnn-binary.py --size "$m_size" \
        --dataset_directory "./dataset" \
        --learning_rate 4e-4 \
        --weight_decay 1e-3 \
        --threshold 0.5 \
        --graph_layer_type "$m_type" \
        --epochs 150 \
        --stochastic_path "$m_sto" \
        --k_neighbours "$m_knn" \
        --batch_size 32 \
        --seed 42 \
        --cnn_extraction_layer "$m_extract" \
        --cnn_res_blocks "$m_res_blocks"
}

run_cnn_gnn_multilabel() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4
    local m_extract=$5
    local m_res_blocks=$6

    python3 cnn-gnn-multilabel.py --size "$m_size" \
        --dataset_directory "./dataset" \
        --learning_rate 1e-5 \
        --weight_decay 1e-3 \
        --threshold 0.5 \
        --graph_layer_type "$m_type" \
        --epochs 150 \
        --stochastic_path "$m_sto" \
        --k_neighbours "$m_knn" \
        --batch_size 32 \
        --seed 42 \
        --cnn_extraction_layer "$m_extract" \
        --cnn_res_blocks "$m_res_blocks"
}

# Examples:
# run_cnn_binary "3,4,6,3"
# run_gnn "GCN" "medium" 7 0.1
# run_cnn_gnn "GCN" "medium" 7 0.1 "layer4" "3,4,6,3"
# run_cnn_gnn_multilabel "GCN" "medium" 7 0.1 "layer4" "3,4,6,3"
# run_gnn "GCN" "medium" 7 0.1

run_cnn_gnn "GCN" "big" 7 0.1 "layer4" "3,4,6,3"
