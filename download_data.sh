#!/bin/bash

mkdir -p data
cd data

curl -L -o mnist-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset

unzip mnist-dataset.zip

rm mnist-dataset.zip
