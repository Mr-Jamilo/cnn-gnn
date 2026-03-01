#!/bin/bash
curl -L -o ./retinal-disease-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/retinal-disease-classification

unzip -o ./retinal-disease-classification.zip -d ./dataset
rm ./retinal-disease-classification.zip

