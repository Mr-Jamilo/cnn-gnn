#!/bin/bash

source .venv/bin/activate

run_cnn_binary() {
    local m_res_blocks=$1

    python3 -m models.binary.cnn \
        --dataset_directory "./dataset" \
        --learning_rate 1e-5 \
        --weight_decay 1e-3 \
        --threshold 0.5 \
        --epochs 150 \
        --batch_size 32 \
        --seed 42 \
        --cnn_res_blocks "$m_res_blocks"
}

run_gnn_binary() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4

    python3 -m models.binary.gnn --size "$m_size" \
        --dataset_directory "./dataset" \
        --learning_rate 1e-5 \
        --weight_decay 1e-3 \
        --threshold 0.5 \
        --graph_layer_type "$m_type" \
        --epochs 150 \
        --stochastic_path "$m_sto" \
        --k_neighbours "$m_knn" \
        --batch_size 32 \
        --seed 42
}

run_cnn_gnn_binary() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4
    local m_extract=$5
    local m_res_blocks=$6

    python3 -m models.binary.cnn-gnn --size "$m_size" \
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

run_cnn_multilabel() {
    local m_res_blocks=$1

    python3 -m models.multilabel.cnn \
        --dataset_directory "./dataset" \
        --learning_rate 1e-4 \
        --weight_decay 1e-3 \
        --threshold 0.5 \
        --epochs 150 \
        --batch_size 32 \
        --seed 42 \
        --cnn_res_blocks "$m_res_blocks"
}

run_cnn_gnn_multilabel() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4
    local m_extract=$5
    local m_res_blocks=$6

    python3 -m models.multilabel.cnn-gnn --size "$m_size" \
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

run_gnn_multilabel() {
    local m_type=$1
    local m_size=$2
    local m_knn=$3
    local m_sto=$4

    python3 -m models.multilabel.gnn --size "$m_size" \
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




# Examples:
# run_cnn_binary "3,4,6,3"
# run_gnn_binary "GCN" "medium" 7 0.1
# run_cnn_gnn_binary "GCN" "medium" 7 0.1 "layer4" "3,4,6,3"
# run_cnn_multilabel "3,4,6,3"
# run_gnn_multilabel "GCN" "medium" 7 0.1
# run_cnn_gnn_multilabel "GCN" "medium" 7 0.1 "layer4" "3,4,6,3"

#run_cnn_binary "2,2,2,2"
#run_gnn_binary "GCN" "small" 3 0.1
run_cnn_gnn_binary "GCN" "small" 9 0.1 "layer4" "3,4,6,3"

#run_cnn_multilabel "2,2,2,2"
#run_gnn_multilabel "GCN" "medium" 5 0.1
#run_cnn_gnn_multilabel "GCN" "medium" 5 0.1 "layer3" "2,2,2,2"


#for x in "GCN" "GAT" "GIN"; do
#  for y in "tiny" "small" "medium" "big"; do
#    for z in 3 5 7 9; do
#      if [[ "$y" == "big" ]]; then
#        run_gnn_multilabel $x $y $z 0.3
#      else
#        run_gnn_multilabel $x $y $z 0.1
#      fi
#   done
#  done
#done

#for x in "GCN" "GAT" "GIN"; do
#  for y in "tiny" "small" "medium" "big"; do
#    for z in 3 5 7 9; do
#      for a in "layer3" "layer4" "avgpool"; do
#        for b in "2,2,2,2" "3,4,6,3"; do
#          if [[ "$y" == "big" ]]; then
#            #run_cnn_gnn_binary $x $y $z 0.3 $a $b
#            #run_cnn_gnn_multilabel $x $y $z 0.3 $a $b
#          else
#            #run_cnn_gnn_binary $x $y $z 0.1 $a $b
#            #run_cnn_gnn_multilabel $x $y $z 0.1 $a $b
#          fi
#        done
#      done
#    done
#  done
#done
