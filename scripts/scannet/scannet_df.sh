#!/bin/bash
# export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=-1
CURR_QUERY=150
# # CURR_AREA=5

# TRAIN
CUDA_VISIBLE_DEVICES=1 python  main_instance_segmentation.py \
general.experiment_name="train" \
general.eval_on_segments=true \
general.train_on_segments=true \
general.checkpoint="base model path" \
data.df=1
# data.df represents the percentage of annotated data to be used.
# It can be set to 1, 5, 10, or 20 to indicate the desired percentage.