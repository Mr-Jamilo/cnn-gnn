#!/bin/bash
curl -L -o ./retinal-disease-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/retinal-disease-classification

unzip -o ./retinal-disease-classification.zip -d ./dataset
rm ./retinal-disease-classification.zip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
pip install torchinfo scikit-learn matplotlib pillow torchmetrics timm
