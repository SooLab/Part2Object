import torch
import numpy as np
import os 
from tqdm import tqdm
import math
from multiprocessing import Pool
import time
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='cluster')
    parser.add_argument('--input', help='processed data dir')
    parser.add_argument('--dataset', default="train", help='train or val')
    parser.add_argument('--output', default="./exp/", help='directory to save the experiment results')

    
    args = parser.parse_args()

    return args

args = parse_args()

# ================ config =================
threshold = 0.05
topk = [0.6, 0.5, 0.4, 0.3, 0.2]
num_layers = 5
b_factor = 3
iou_threshold = 0.6
# ================ config =================


# ================ file path =================
processed_data_dir = args.input
dataset = args.dataset # or val
# ================ file path =================


data_dir = f"{processed_data_dir}processed" 
scene_list = sorted(os.listdir(f"{processed_data_dir}processed/{dataset}")) # can set to train

feat_dir = f"{processed_data_dir}DINO_point_feats"
distance_martrix_dir = f"{processed_data_dir}dis_matrixes_initseg"
superpoints_dir = f"{processed_data_dir}initial_superpoints"
bbox_dir = f"{processed_data_dir}bbox_prior"
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
   

def box_intersection(box1, box2):
    intersect_x = max(0, min(box1[3], box2[3]) - max(box1[0], box2[0]))
    intersect_y = max(0, min(box1[4], box2[4]) - max(box1[1], box2[1]))
    intersect_z = max(0, min(box1[5], box2[5]) - max(box1[2], box2[2]))
    return intersect_x * intersect_y * intersect_z

def box_union(box1, box2):
    volume_box1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    volume_box2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    return volume_box1 + volume_box2 - box_intersection(box1, box2)

def box_iou(box1, box2):
    intersection = box_intersection(box1, box2)
    union = box_union(box1, box2)
    iou = intersection / union if union > 0 else 0
    return iou

def batch_box_iou(box, boxes):
    box_min = np.broadcast_to(box[:, :3], (len(boxes), 3))
    box_max = np.broadcast_to(box[:, 3:], (len(boxes), 3))

    intersect_mins = np.maximum(boxes[:, :3], box_min)
    intersect_maxes = np.minimum(boxes[:, 3:], box_max)

    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0)
    intersection = intersect_wh.prod(axis=1)

    volume_boxes = np.prod(boxes[:, 3:] - boxes[:, :3], axis=1)
    volume_box = np.prod(box[:, 3:] - box[:, :3])
    union = volume_boxes + volume_box - intersection
    
    iou = intersection / union
    return iou


def find_big_than_th_similarities(similarity_matrix, th):
    flattened_matrix = similarity_matrix.flatten()
    top_k_indices = np.argsort(flattened_matrix)

    top_k_pairs = [(i // similarity_matrix.shape[0], i % similarity_matrix.shape[0]) for i in top_k_indices]
    return top_k_pairs

def find_top_k_similarities(similarity_matrix, k):
    flattened_matrix = similarity_matrix.flatten()
    top_k_indices = np.argsort(flattened_matrix)[-k:]
    top_k_pairs = [(i // similarity_matrix.shape[0], i % similarity_matrix.shape[0]) for i in top_k_indices]
    return top_k_pairs

def update_dis_matrix(i, j, distance_matrix):
    distance_1 = distance_matrix[i]
    distance_2 = distance_matrix[j]
    distance_new = np.minimum(distance_1, distance_2)
    distance_new = np.delete(distance_new, j)

    new_distance_matrix = np.delete(distance_matrix, j, axis=0)  
    new_distance_matrix = np.delete(new_distance_matrix, j, axis=1)
    
    if i < j:
        new_distance_matrix[i, :] = distance_new
        new_distance_matrix[:, i] = distance_new
    else:
        new_distance_matrix[i-1, :] = distance_new
        new_distance_matrix[:, i-1] = distance_new
    return new_distance_matrix


def get_center(cube):
    x1, y1, z1 = cube[0]
    x2, y2, z2 = cube[1]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_z = (z1 + z2) / 2
    return center_x, center_y, center_z

def get_diagonal_length(cube):
    x1, y1, z1 = cube[0]
    x2, y2, z2 = cube[1]
    length = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
    return length

def naive_distance(super_point_A, super_point_B):

    distances = np.sqrt(np.sum((super_point_A[:, np.newaxis] - super_point_B) ** 2, axis=2))

    min_distances = np.min(distances, axis=1)

    min_distance = np.min(min_distances)
    return min_distance

def bbox_distance(super_point_A, super_point_B):
    x1, y2, z2 = np.min(super_point_A, axis=0) 
    x2, y1, z1 = np.max(super_point_A, axis=0)

    x3, y4, z4 = np.min(super_point_B, axis=0)
    x4, y3, z3 = np.max(super_point_B, axis=0)

    cube1 = [[x1, y1, z1], [x2, y2, z2]]
    cube2 = [[x3, y3, z3], [x4, y4, z4]]
    cube1 = np.array(cube1)
    cube2 = np.array(cube2)

    center1 = get_center(cube1)
    center2 = get_center(cube2)

    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2 + (center1[2] - center2[2])**2)**0.5

    diagonal_length1 = get_diagonal_length(cube1) / 2
    diagonal_length2 = get_diagonal_length(cube2) / 2
    total_diagonal_length = diagonal_length1 + diagonal_length2

    if distance <= total_diagonal_length:
        return naive_distance(super_point_A, super_point_B)
    else:
        return distance

def calculate_distance(pair):
    i, j, super_points, super_points_ids, coords = pair
    mask_A = super_points == super_points_ids[i]
    mask_B = super_points == super_points_ids[j]
    distance = bbox_distance(coords[mask_A], coords[mask_B])

    return (i, j, distance)

for scan in tqdm(scene_list):
    if ".pth" in scan:
        scan = scan.split(".")[0]
    is_scan_finished = 1
    for layer in range(num_layers):
        if not os.path.exists(os.path.join(output_dir,f"{scan}_layer{layer}_scores.pth")):
            is_scan_finished = 0
            break
    if is_scan_finished:
        continue
    
    not_marge = []
    layer_score = [0] * num_layers

    feats = torch.load(os.path.join(feat_dir,f"{scan}.pth"))
    distance_matrix = torch.load(os.path.join(distance_martrix_dir,f"{scan}.pth"))
    super_points = np.load(os.path.join(superpoints_dir,f"{scan}_superpoint.npy"))

    try:
        points_path = f"{data_dir}/val/{scan}.pth"
        points = torch.load(points_path)
    except:
        points_path = f"{data_dir}/train/{scan}.pth"
        points = torch.load(points_path)

    coords = points["coord"]
    normal = points["normal"]
    
    try:
        mask_cut = torch.load(f"{bbox_dir}/{scan}_merge_ncut.pth")
        boxes = np.zeros((len(np.unique(mask_cut)) - 1, 6))
        for i, b in enumerate(np.unique(mask_cut)[1:]):
            boxes[i, :3] = np.min(coords[mask_cut == b], axis=0)
            boxes[i, 3:] = np.max(coords[mask_cut == b], axis=0)
    except:
        boxes = np.zeros((len(np.unique(mask_cut)) - 1, 6))


    for layer in range(num_layers):

        super_points_ids_left = []
        super_points_feat = []
        super_points_ids = np.unique(super_points)
        super_points_ids = super_points_ids[super_points_ids != -2]

        keep = [i for i in range(super_points_ids.shape[0])]

        mask1 = np.zeros((super_points.shape[0], super_points_ids.max()+1))
        mask1[np.arange(super_points.shape[0]), super_points] = 1
        mask1 = mask1[:, super_points_ids]
        mask1 = mask1.T

        mask2 = (feats == 0).sum(1) != 768
        mask2 = np.repeat(mask2.reshape(1, super_points.shape[0]), super_points_ids.shape[0], axis=0)

        super_point_mask = np.logical_and(mask1, mask2)

        num_points = super_point_mask.sum(axis=1)
        
        if layer == 0:
            zero_points = num_points == 0
            zero_super_points_ids = super_points_ids[zero_points]
            zero_mask = np.in1d(super_points, zero_super_points_ids)

            super_points[zero_mask] = -2
            keep = np.array(keep)
            keep = keep[~zero_points]
            super_points_ids = super_points_ids[~zero_points]
            super_point_mask = super_point_mask[~zero_points, :]
            
        
        feats = feats + 1e-6
        super_point_feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        super_point_feats_mean = np.dot(super_point_mask, super_point_feats) / np.sum(super_point_mask, axis=1, keepdims=True)
        
        self_sim = super_point_feats_mean @ super_point_feats.T
        self_sim = self_sim * super_point_mask

        super_points_feat = np.dot(self_sim, super_point_feats) / np.sum(self_sim, axis=1, keepdims=True)

        distance_matrix = distance_matrix[keep][:, keep]

        sim_matrix = super_points_feat @ super_points_feat.T

        dis_mask = distance_matrix > threshold
        self_mask = np.eye(sim_matrix.shape[0], dtype=bool)
        tri_mask = np.tri(sim_matrix.shape[0], sim_matrix.shape[0], k=-1, dtype=bool)
        sim_matrix[dis_mask] = -np.inf
        sim_matrix[self_mask] = -np.inf
        sim_matrix[tri_mask] = -np.inf

        num_topk = math.floor(topk[layer]*sim_matrix.shape[0])
        topk_pair = find_top_k_similarities(sim_matrix, num_topk)

        id_in_dis = np.arange(distance_matrix.shape[0])
        dis_copy = distance_matrix.copy()
        update_super_points_ids = super_points_ids.copy()
        

        child_labels = np.zeros(distance_matrix.shape[0]) - 2


        for pair in topk_pair:
            marge_i, marge_j = pair
                
            ori_label = super_points_ids[marge_i]
            to_marge_label = super_points_ids[marge_j]

            marge_mask = super_points == ori_label
            to_marge_mask = super_points == to_marge_label

            coord1 = coords[marge_mask].mean(axis = 0)
            normal1 = normal[marge_mask].mean(axis=0)

            coord2 = coords[to_marge_mask].mean(axis = 0)
            normal2 = normal[to_marge_mask].mean(axis=0)


            if ori_label != to_marge_label and ori_label not in not_marge:


                if child_labels[marge_i] * b_factor > to_marge_mask.sum() or child_labels[marge_i] == -2:
                    super_points[to_marge_mask] = ori_label
                    super_points_ids[super_points_ids == super_points_ids[marge_j]] = ori_label

                    child_labels[marge_i] = to_marge_mask.sum()
                    distance_matrix = update_dis_matrix(id_in_dis[marge_i], id_in_dis[marge_j], distance_matrix)

                    id_in_dis[id_in_dis >= id_in_dis[marge_j]] -= 1 
                    id_in_dis[super_points_ids == super_points_ids[marge_j]] = id_in_dis[marge_i]

                    marged_mask = super_points == ori_label
                    marged_coord = coords[marged_mask]
                    marged_bbox = np.concatenate((np.min(marged_coord, axis=0), np.max(marged_coord, axis=0)), axis=0).reshape(1, 6)


                    ious = batch_box_iou(marged_bbox, boxes)
                    if ious.max() > iou_threshold:
                        not_marge.append(ori_label)
                        layer_score[layer] += 1
                            
        super_points_save = super_points.copy()
        torch.save(super_points, os.path.join(output_dir,f"{scan}_layer{layer}_scores.pth"))