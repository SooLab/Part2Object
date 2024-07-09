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
general.train_on_segments=true



# # TEST
CUDA_VISIBLE_DEVICES=0 python  main_instance_segmentation.py \
general.experiment_name="test" \
general.project_name="scannet_eval" \
general.checkpoint='path of ckpt' \
data/datasets=scannet \
general.train_mode=false \
general.eval_on_segments=true \
general.train_on_segments=true \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
