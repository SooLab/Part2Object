import os
import numpy as np

from tqdm import tqdm
import torch

from multiprocessing import Pool

def naive_distance(super_point_A, super_point_B):

    distances = np.sqrt(np.sum((super_point_A[:, np.newaxis] - super_point_B) ** 2, axis=2))

    min_distances = np.min(distances, axis=1)

    min_distance = np.min(min_distances)
    return min_distance

def calculate_distance(pair):
    i, j, super_points, super_points_ids, coords = pair
    mask_A = super_points == super_points_ids[i]
    mask_B = super_points == super_points_ids[j]
    distance = naive_distance(coords[mask_A], coords[mask_B])

    return (i, j, distance)


scene_list = sorted(os.listdir("data/frames_square/"))

for scan in tqdm(scene_list):

    try:
        points_path = f"data/val/{scan}.pth"
        points = torch.load(points_path)
    except:
        points_path = f"data/train/{scan}.pth"
        points = torch.load(points_path)

    superpoints_path = f"init_segments/{scan}.pth"
    super_points = torch.load(superpoints_path)

    super_points_ids = np.unique(super_points)

    coords = points["coord"]
    colors = points["color"]


    super_point_coors = []
    super_point_colors = []
    super_point_feats = []

    distance_matrix = np.zeros((super_points_ids.shape[0], super_points_ids.shape[0]))

    with Pool() as pool:
        pairs = [(i, j, super_points, super_points_ids, coords) for i in range(super_points_ids.shape[0]) for j in range(i+1, super_points_ids.shape[0])]
        results = pool.map(calculate_distance, pairs)

    for result in results:
        i, j, distance = result
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    torch.save(distance_matrix, f"dis_matrixes_initseg/{scan}.pth")