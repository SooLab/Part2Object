import torch
import numpy as np
import os 
from tqdm import tqdm
import math
from multiprocessing import Pool
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='cluster')
    parser.add_argument('--input', help='processed data dir')
    parser.add_argument('--dataset', default="train", help='train or val')
    parser.add_argument('--output', help='directory to save the experiment results')
    parser.add_argument('--output-processed', 
                        help='directory to save the experiment results after post processing')
    args = parser.parse_args()

    return args

args = parse_args()


# ================ file path =================
processed_data_dir = args.input
dataset = args.dataset # or val
# ================ file path =================

# ================ config =================
small_ = 500
normal_ = 750
large_ = 10000

config = dict(
    threshold = 0.05,
    num_layers = 1,
    small_threshold = small_,
    normal_threshold = normal_,
    large_threshold = large_,

    superpoints_dir = args.output, # path of result produced by topk_merge.py
    scene_list = sorted(os.listdir(f"{processed_data_dir}data/{dataset}")),
    pcd_dir = f"{processed_data_dir}data"  ,
    output_dir = args.output_processed, # path of result produced by this script
    post_fix = "_layer4_scores.pth",
)
# ================ config =================

this_config = config
threshold = this_config["threshold"]
num_layers = this_config["num_layers"]
small_threshold = this_config["small_threshold"]
normal_threshold = this_config["normal_threshold"]
large_threshold = this_config["large_threshold"]
superpoints_dir = this_config["superpoints_dir"]
scene_list = this_config["scene_list"]
pcd_dir = this_config["pcd_dir"]
output_dir = this_config["output_dir"]
post_fix = this_config["post_fix"]


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
    # super_point_A:(N, 3)
    # super_point_B:(M, 3)

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
        scan = scan.replace(".pth","")
    if not "scene" in scan:
        scan = f"scene{scan}"

    superpoints_path = os.path.join(superpoints_dir, scan + post_fix)
    super_points = torch.load(superpoints_path)

    try:
        points_path = os.path.join(pcd_dir,"val",f"{scan}.pth")
        points = torch.load(points_path)
    except:
        points_path = os.path.join(pcd_dir,"train",f"{scan}.pth")
        points = torch.load(points_path)

    coords = points["coord"]
    
    for layer in range(num_layers):
        
        output_path = os.path.join(output_dir, f"{scan}_layer0.pth")
        if os.path.exists(output_path):
            continue

        super_point_masks = []
        
        super_points_ids = np.unique(super_points)
        if super_points_ids[0] == -2:
            super_points_ids = super_points_ids[1:]
        
        distance_matrix = np.zeros((super_points_ids.shape[0], super_points_ids.shape[0]))

        with Pool() as pool:
            pairs = [(i, j, super_points, super_points_ids, coords) for i in range(super_points_ids.shape[0]) for j in range(i+1, super_points_ids.shape[0])]
            results = pool.map(calculate_distance, pairs)

        for result in results:
            i, j, distance = result
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

        self_mask = np.eye(distance_matrix.shape[0], dtype=bool)
       
        distance_matrix[self_mask] = np.inf


        for s, super_points_id in enumerate(super_points_ids):

            mask1 = super_points == super_points_id
            nabor_index = np.argmin(distance_matrix[s])
            nabor_id = super_points_ids[nabor_index]
            mask_nabor = super_points == nabor_id

            if distance_matrix[s, nabor_index] > threshold:
                continue

            aera_mask = mask1.sum()
            aera_nabor = mask_nabor.sum()
            if aera_mask < small_threshold:
                super_points[mask1] = nabor_id
                super_points_ids[s] = nabor_id
            elif aera_mask < normal_threshold and aera_nabor > large_threshold:
                super_points[mask1] = nabor_id
                super_points_ids[s] = nabor_id

        torch.save(super_points, output_path)