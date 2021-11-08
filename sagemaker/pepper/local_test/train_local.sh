#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

sudo docker run  --gpus all -v "$(pwd)/test_dir:/opt/ml" --rm ${image} train_net --config-file configs/centermask/centermask_V_39_eSE_FPN_ms_3x_cuc.yaml

