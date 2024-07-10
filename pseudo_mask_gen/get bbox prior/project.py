import os
import sys
# import h5py
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from imageio import imread
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from collections import Counter

# sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
# from lib.config import CONF
from projection import ProjectionHelper
# from lib.enet import create_enet_for_3d
import copy
import yaml


# SCANNET_LIST = CONF.SCANNETV2_LIST
SCANNET_DATA = '/remote-home/share/Datasets/Scannetv2/'
SCANNET_FRAME_ROOT = '/remote-home/share/Datasets/Scannetv2/frames_square'
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, "{}") # name of the file

# ENET_FEATURE_PATH = CONF.ENET_FEATURES_PATH
# ENET_FEATURE_DATABASE = CONF.MULTIVIEW

# projection
INTRINSICS = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0][0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1][1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0][2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1][2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic

INTRINSICS = adjust_intrinsic(INTRINSICS, (41, 32), (320, 240))
PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [320, 240], 0.05)

# ENET_PATH = CONF.ENET_WEIGHTS
ENET_GT_PATH = SCANNET_FRAME_PATH

# NYU40_LABELS = CONF.NYU40_LABELS
SCANNET_LABELS = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']

# PC_LABEL_ROOT = os.path.join(CONF.PATH.OUTPUT, "projections")
# PC_LABEL_PATH = os.path.join(PC_LABEL_ROOT, "{}.ply")

# def get_nyu40_labels():
#     labels = ["unannotated"]
#     labels += pd.read_csv(NYU40_LABELS)["nyu40class"].tolist()
    
#     return labels

# def get_prediction_to_raw():
#     labels = get_nyu40_labels()
#     mapping = {i: label for i, label in enumerate(labels)}

#     return mapping

# def get_nyu_to_scannet():
#     nyu_idx_to_nyu_label = get_prediction_to_raw()
#     scannet_label_to_scannet_idx = {label: i for i, label in enumerate(SCANNET_LABELS)}

#     # mapping
#     nyu_to_scannet = {}
#     for nyu_idx in range(41):
#         nyu_label = nyu_idx_to_nyu_label[nyu_idx]
#         if nyu_label in scannet_label_to_scannet_idx.keys():
#             scannet_idx = scannet_label_to_scannet_idx[nyu_label]
#         else:
#             scannet_idx = 0
#         nyu_to_scannet[nyu_idx] = scannet_idx

#     return nyu_to_scannet

# def create_color_palette():
#     return {
#         "unannotated": (0, 0, 0),
#         "floor": (152, 223, 138),
#         "wall": (174, 199, 232),
#         "cabinet": (31, 119, 180),
#         "bed": (255, 187, 120),
#         "chair": (188, 189, 34),
#         "sofa": (140, 86, 75),
#         "table": (255, 152, 150),
#         "door": (214, 39, 40),
#         "window": (197, 176, 213),
#         "bookshelf": (148, 103, 189),
#         "picture": (196, 156, 148),
#         "counter": (23, 190, 207),
#         "desk": (247, 182, 210),
#         "curtain": (219, 219, 141),
#         "refridgerator": (255, 127, 14),
#         "bathtub": (227, 119, 194),
#         "shower curtain": (158, 218, 229),
#         "toilet": (44, 160, 44),
#         "sink": (112, 128, 144),
#         "otherfurniture": (82, 84, 163),
#     }

# def get_scene_list(args):
#     if args.scene_id == "-1":
#         with open(SCANNET_LIST, 'r') as f:
#             return sorted(list(set(f.read().splitlines())))
#     else:
#         return [args.scene_id]

def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = 0
    
    return result

def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]

def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if torch.all(group_ids == 0):
        return np.array(group_ids)
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != 0])
    try:
        mapping = np.full(np.max(unique_values) + 2, 0)
    except:
        print(group_ids)
        print(unique_values)
        exit()
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array

# def cal_group(label_list, index, ratio=0.4):
#     if len(index) == 1:
#         return(label_list[index[0]])
#     group_0 = label_list[index[0]]
#     group_1 = label_list[index[1]]
    
#     group_1[group_1 != 0] += group_0.max() + 1
    
#     unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
#     group_0_counts = dict(zip(unique_groups, group_0_counts))
#     unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
#     group_1_counts = dict(zip(unique_groups, group_1_counts))

#     # Calculate the group number correspondence of overlapping points
#     group_overlap = {}
#     for i in range(len(group_0)):
#         group_i = group_1[i]
#         group_j = group_0[i]
#         if group_i == 0:
#             group_1[i] = group_0[i]
#             continue
#         if group_j == 0:
#             continue
#         if group_i not in group_overlap:
#             group_overlap[group_i] = {}
#         if group_j not in group_overlap[group_i]:
#             group_overlap[group_i][group_j] = 0
#         group_overlap[group_i][group_j] += 1

#     # Update group information for point cloud 1
#     for group_i, overlap_count in group_overlap.items():
#         # for group_j, count in overlap_count.items():
#         max_index = np.argmax(np.array(list(overlap_count.values())))
#         group_j = list(overlap_count.keys())[max_index]
#         count = list(overlap_count.values())[max_index]
#         try:
#             total_count = min(group_0_counts[group_j.item()], group_1_counts[group_i.item()]).astype(np.float32)
#         except:
            
            
#             print(group_0_counts.keys())
#             print(group_j.item())
#             print(group_1_counts.keys())
#             print(group_i.item())
#             exit()
#         # print(count / total_count)
#         if count / total_count >= ratio:
#             group_1[group_1 == group_i] = group_j
#     return group_1

# 3 case for group1 and group 0: 
# 1. 0,0 continue 
# 2. 0,label 0->label  
# 3. label1,label2 find max count label in label2 , if count/total_count > ratio, label1->label2
def cal_group(label_list, index, coord, ratio=0.5):
    if len(index) == 1:
        # return label_list[index[0]], feat_list[index[0]]
        return label_list[index[0]]

    group_0 = label_list[index[0]]
    group_1 = label_list[index[1]]
    
    # feats_0 = feat_list[index[0]]
    # feats_1 = feat_list[index[1]]
    
    
    group_1[group_1 != 0] += group_0.max() + 1
    
    unique_groups_0, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups_0, group_0_counts))
    unique_groups_1, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups_1, group_1_counts))
    
    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i in range(len(group_0)):
        group_i = group_1[i]
        group_j = group_0[i]
        if group_i == 0:
            group_1[i] = group_0[i]
            continue
        if group_j == 0:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        
        # print(feats_1.shape)
        
        
        # mean_i = torch.where(torch.tensor(group_1).unsqueeze(-1).repeat(1, 768)==group_i, feats_1, zero_map)
        # mean_i = feats_1[torch.tensor(group_1)==group_i, :].mean(dim=0)
        # print(feats_1[torch.tensor(group_1)==group_i, :].shape)
        # print(mean_i.shape)
        # exit()
        # mean_i = mean_i / mean_i.norm(keepdim=True)
        
        # mean_j = torch.where(torch.tensor(group_0).unsqueeze(-1).repeat(1, 768)==group_j, feats_0, zero_map)
        # mean_j = feats_0[torch.tensor(group_0)==group_j, :].mean(dim=0)
        # mean_j = mean_j / mean_j.norm(keepdim=True)
        
        
        
        try:
            total_count = min(group_0_counts[group_j.item()], group_1_counts[group_i.item()]).astype(np.float32)
            # total_count = group_0_counts[group_j.item()] + group_1_counts[group_i.item()] - count
        except:
            
            
            print(group_0_counts.keys())
            print(group_j.item())
            print(group_1_counts.keys())
            print(group_i.item())
            exit()
        # print(count / total_count)
        # print(mean_j @ mean_j)
        # if count / total_count >= ratio or mean_i @ mean_j >= 0.8:
        if count / total_count >= ratio:
            
            group_1[group_1 == group_i] = group_j
            
    # feats_0_mask = ((feats_0 == 0).sum(1) != 768).bool()
    # feats_1_mask = ((feats_1 == 0).sum(1) == 768).bool()
    
    # mask = feats_0_mask * feats_1_mask
    # feats_1[mask] = feats_0[mask]
    
    # mask = ~feats_1_mask * feats_0_mask
    # feats_1[mask] = torch.max(feats_1[mask], feats_0[mask])
    
    # return group_1, feats_1
    return group_1


def to_tensor(arr):
    # return torch.Tensor(arr).cuda()
    return torch.Tensor(arr)
    

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    
    return image

def load_image(file, image_dims):
    # print(file)
    image = imread(file)
    # preprocess
    image = resize_crop_image(image, image_dims)
    if len(image.shape) == 3: # color image
        image =  np.transpose(image, [2, 0, 1])  # move feature to front
        image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
    elif len(image.shape) == 2: # label image
#         image = np.expand_dims(image, 0)
        pass
    else:
        raise
        
    return image

def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]

    return np.asarray(lines).astype(np.float32)

def load_depth(file, image_dims):
    depth_image = imread(file)
    # preprocess
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0

    return depth_image

# def visualize(coords, labels):
#     palette = create_color_palette()
#     nyu_to_scannet = get_nyu_to_scannet()
#     vertex = []
#     for i in range(coords.shape[0]):
#         vertex.append(
#             (
#                 coords[i][0],
#                 coords[i][1],
#                 coords[i][2],
#                 palette[SCANNET_LABELS[nyu_to_scannet[labels[i]]]][0],
#                 palette[SCANNET_LABELS[nyu_to_scannet[labels[i]]]][1],
#                 palette[SCANNET_LABELS[nyu_to_scannet[labels[i]]]][2]
#             )
#         )
    
#     vertex = np.array(
#         vertex,
#         dtype=[
#             ("x", np.dtype("float32")), 
#             ("y", np.dtype("float32")), 
#             ("z", np.dtype("float32")),
#             ("red", np.dtype("uint8")),
#             ("green", np.dtype("uint8")),
#             ("blue", np.dtype("uint8"))
#         ]
#     )

#     output_pc = PlyElement.describe(vertex, "vertex")
#     output_pc = PlyData([output_pc])
#     os.makedirs(PC_LABEL_ROOT, exist_ok=True)
#     output_pc.write(PC_LABEL_PATH.format(args.scene_id))

def get_scene_data(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        # scene_data[scene_id] = {}
        scene_data[scene_id] = torch.load(os.path.join(SCANNET_DATA, "val", scene_id)+".pth")["coord"]
    
    return scene_data

def compute_projection(points, depth, camera_to_world):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        
        :return indices_3d (array with point indices that correspond to a pixel),
        :return indices_2d (array with pixel indices that correspond to a point)

        note:
            the first digit of indices represents the number of relevant points
            the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros(num_frames, num_points + 1).long()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(to_tensor(points), to_tensor(depth[i]), to_tensor(camera_to_world[i]))
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()
        
    return indices_3ds, indices_2ds

# def create_enet():
#     enet_fixed, enet_trainable, enet_classifier = create_enet_for_3d(41, ENET_PATH, 21)
#     enet = nn.Sequential(
#         enet_fixed,
#         enet_trainable,
#         enet_classifier
#     ).cuda()
#     enet.eval()
#     for param in enet.parameters():
#         param.requires_grad = False

#     return enet

def find_surround_point(center,pcd_coordinate,length):
    min_xzy = center - length
    max_xzy = center + length

    upper_idx = (
                np.sum((pcd_coordinate[:, :3] <= max_xzy).astype(np.int32), 1) == 3
            )
    lower_idx = (
        np.sum((pcd_coordinate[:, :3] >= min_xzy).astype(np.int32), 1) == 3
    )

    new_pointidx = (upper_idx) & (lower_idx)
    return new_pointidx

def find_pure_seg(label,coord):
    seg_id,count = np.unique(label,return_counts=True)
    id_count = {}
    for id in seg_id:
        if id == 0:
            continue
        else:
            center = coord[label == id].mean(0)
            range = (np.max(coord[label == id,:3],axis = 0)-np.min(coord[label == id,:3],axis = 0))/2
            surrent_point_id = find_surround_point(center,coord,range)        
            id_count[id] = (np.sum(label[surrent_point_id] == id) / np.sum(surrent_point_id))
    # print(id_count)  
    for id in seg_id:
        if id == 0:
            continue
        if id_count[id] < 0.3:
                label[label == id] = 0
    return label
    # find max top 3 value in id_count, return the key
    # id_count = sorted(id_count.items(), key=lambda item:item[1], reverse=True)
            
def cal_instance(labels, coords, ratio=0.5):
    instance_dict = {}            

def compute_bbox(coord,index):
    return (np.max(coord[index,:],axis = 0),np.min(coord[index,:3],axis = 0))

def _save_yaml(path, file):
    with open(path, "w") as f:
        yaml.safe_dump(
            file, f, default_style=None, default_flow_style=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, default="-1")
    parser.add_argument("--gt", action="store_false")
    parser.add_argument("--maxpool", action="store_false", help="use max pooling to aggregate features \
         (use majority voting in label projection mode)")
    args = parser.parse_args()

    # dinov2_vitb14 = torch.hub.load('/root/dinov2', 'dinov2_vitb14', source='local').cuda()
    # scene_list = get_scene_list(args)
    # scene_list = sorted(os.listdir("results8_2d"))
    # scene_list = ["scene0000_00", "scene0000_01", "scene0000_02", "scene0001_00", "scene0137_00", "scene0138_00", "scene0139_00", "scene0140_00", "scene0141_00", "scene0282_00", "scene0283_00", "scene0284_00", "scene0285_00", "scene0286_00", "scene0287_00", "scene0288_00", "scene0425_00", "scene0426_00", "scene0427_00", "scene0428_00", "scene0429_00", "scene0430_00", "scene0575_00", "scene0576_00"]

    # scene_list = ["scene0000_00","scene0011_00","scene0015_00","scene0018_00","scene0030_00","scene0092_01",
    #               "scene0101_00","scene0123_00","scene0194_00","scene0302_00"
    #               ] 
    # scene_list = ["scene0138_00", "scene0140_00", 
    #               "scene0141_00", "scene0282_00", "scene0283_00", 
    #               "scene0284_00", "scene0285_00", "scene0286_00", 
    #               "scene0287_00", "scene0288_00", "scene0425_00",]
    scene_list = os.listdir("outputs")
    # scene_list = ["scene0000_00"]
    # scene_data = get_scene_data(scene_list)
    # enet = create_enet()
    for scene_id in tqdm(scene_list):
        if not "scene" in scene_id:
            continue
        # if os.path.exists(f"/remote-home/yangbin/CutLER/maskcut/project_label/{scene_id}.txt"):
        #     continue
        # if os.path.exists(os.path.join("proj5", scene_id + ".pth")):
        #     continue

        # scene_id = scene_id[:-4]
        # scene_id = "scene0000_00"
        # scene = scene_data[scene_id]
        try:
            scene = torch.load(os.path.join(SCANNET_DATA, "train", scene_id)+".pth")["coord"]
        except:
            scene = torch.load(os.path.join(SCANNET_DATA, "val", scene_id)+".pth")["coord"]

        # scene = torch.load(f"superpoints2/{scene_id}.pth")
        # load frames
        # frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir(SCANNET_FRAME_PATH.format(scene_id) + "/color"))))
        frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir("outputs/{}".format(scene_id)))))
        scene_images = np.zeros((len(frame_list), 3, 256, 328))
        scene_depths = np.zeros((len(frame_list), 240, 320))
        scene_poses = np.zeros((len(frame_list), 4, 4))
        for i, frame_id in enumerate(frame_list):
            scene_images[i] = load_image(SCANNET_FRAME_PATH.format(scene_id) + "/color" + "/{}.jpg".format(frame_id), [328, 256])
            scene_depths[i] = load_depth(SCANNET_FRAME_PATH.format(scene_id) + "/depth" + "/{}.png".format(frame_id), [320, 240])
            scene_poses[i] = load_pose(SCANNET_FRAME_PATH.format(scene_id) + "/pose" + "/{}.txt".format(frame_id))

        # compute projections for each chunk
        projection_3d, projection_2d = compute_projection(scene, scene_depths, scene_poses)
        
        # compute valid projections
        projections = []
        for i in range(projection_3d.shape[0]):
            num_valid = projection_3d[i, 0]
            if num_valid == 0:
                continue

            projections.append((frame_list[i], projection_3d[i], projection_2d[i]))

        # # project
        # labels = None
        # for i, projection in enumerate(projections):
        #     frame_id = projection[0]
        #     projection_3d = projection[1]
        #     projection_2d = projection[2]
        #     if args.gt:
        #         feat = to_tensor(load_image(ENET_GT_PATH.format(scene_id, "labelv2", "{}.png".format(frame_id)), [41, 32])).unsqueeze(0)
        #     else:
        #         image = load_image(SCANNET_FRAME_PATH.format(scene_id, "color", "{}.jpg".format(frame_id)), [328, 256])
        #         feat = enet(to_tensor(image).unsqueeze(0)).max(1)[1].unsqueeze(1)

        #     proj_label = PROJECTOR.project(feat, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0)
        #     if i == 0:
        #         labels = proj_label
        #     else:
        #         labels[labels == 0] = proj_label[labels == 0]

        # project
        # labels = to_tensor(scene).new(scene.shape[0], len(projections)).fill_(0).float()
        labels = []
        # feats = []

        for i, projection in enumerate(projections):
            frame_id = projection[0]
            projection_3d = projection[1]
            projection_2d = projection[2]
            
            if args.gt:
                # feat = to_tensor(load_image(ENET_GT_PATH.format(scene_id) + "/labelv2" + "/{}.png".format(frame_id), [41, 32])).unsqueeze(0)
                # feat = to_tensor(load_image("scene0000_00/{}".format(scene_id) + "/{}.png".format(frame_id), [320, 240])).unsqueeze(0)
                feat = to_tensor(load_image(f"outputs/{scene_id}/" + "/{}.png".format(frame_id), [320, 240])).unsqueeze(0)
                label,count = feat.unique(return_counts=True)
                mask_label = count > (320*240*0.6)
                del_label = label[mask_label]
                feat[torch.isin(feat,del_label)] = 0
                # if frame_id % 40 == 0:
                #     continue

                # feat = to_tensor(load_image("objmask/" + "/{}.pth.png".format(frame_id), [320, 240])).unsqueeze(0)

                # image = to_tensor(load_image("data/frames_square/{}/color".format(scene_id) + "/{}.jpg".format(frame_id), [322, 238])).unsqueeze(0)
                
                # feat2 = dinov2_vitb14(image).reshape((17, 23, 768)).permute(2, 0, 1).unsqueeze(0)
                # feat2 = torch.nn.functional.interpolate(feat2, (240, 320)).squeeze(0)
            else:
                image = load_image(SCANNET_FRAME_PATH.format(scene_id) + "/color" + "/{}.jpg".format(frame_id), [328, 256])
                # feat = enet(to_tensor(image).unsqueeze(0)).max(1)[1].unsqueeze(1)

            proj_label = PROJECTOR.project(feat, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0) # num_points, 1
            # proj_feat = PROJECTOR.project(feat2, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0) # num_points, 1


            # if args.maxpool:
            #     # only apply max pooling on the overlapping points
            #     # find out the points that are covered in projection
            #     feat_mask = ((proj_label == 0).sum(1) != 1).bool()
            #     # find out the points that are not filled with labels
            #     point_mask = ((labels == 0).sum(1) == len(projections)).bool()

            #     # for the points that are not filled with features
            #     # and are covered in projection, 
            #     # simply fill those points with labels
            #     mask = point_mask * feat_mask
            #     labels[mask, i] = proj_label[mask, 0]

            #     # for the points that have already been filled with features
            #     # and are covered in projection, 
            #     # simply fill those points with labels
            #     mask = ~point_mask * feat_mask
            #     labels[mask, i] = proj_label[mask, 0]
            # else:
            #     if i == 0:
            #         labels = proj_label
            #     else:
            #         labels[labels == 0] = proj_label[labels == 0]

        # aggregate
        # if args.maxpool:
        #     new_labels = []
        #     for label_id in range(labels.shape[0]):
        #         point_label = labels[label_id].cpu().numpy().tolist()
        #         count = dict(Counter(point_label))
        #         count = sorted(count.items(), key=lambda x: x[1], reverse=True)
        #         count = [c for c in count if c[0] != 0]
        #         if count:
        #             new_labels.append(count[0][0])
        #         else:
        #             new_labels.append(0)

            f_labels = torch.FloatTensor(np.array(proj_label.cpu())[:, np.newaxis]).reshape(proj_label.shape[0],).int()
            labels.append(f_labels)
            
            # labels.append(num_to_natural(f_labels))
            # feats.append(proj_feat.cpu())

            # print(labels.shape)
            # torch.save(labels.reshape(labels.shape[0],), "new_proj.pth")
        
        # count = 0
        # for label in labels:
        #     print(np.unique(label))
        #     count += (len(np.unique(label)) - 1)
        # print(count)
        
        import pickle
        file1 = open(f"/remote-home/yangbin/CutLER/maskcut/project_label/{scene_id}.txt", "wb")
        pickle.dump(labels, file1)
        file1.close()
        
        # """========================== bbox =========================="""
        # import time
        # time1 = time.time()
        # print("start merge seg ----")
        # instances_list = [] # [[image_index,object_index]]
        
        # # init label
        # for i in range(len(labels)-1):     
        #     labels[i+1][labels[i+1] != 0] = labels[i].max() + 1
        
        # # init instances_list
        # for seg_id in np.unique(labels[0]):
        #     if seg_id == 0:
        #         continue
        #     instance = [[0,seg_id,compute_bbox(scene,labels[0] == seg_id)]]
        #     instances_list.append(instance)
        
        # # loop all scene, loop all seg
        # for i in range(1,len(labels)):
        #     for seg_id in np.unique(labels[i]):
        #         if seg_id != 0:
        #             continue
                
        #         # merge seg
        #         iou_dict = {} # i:iou
        #         new_mask = labels[i][labels[i] == seg_id]
        #         for j in range(len(instances_list)):              
        #             for shot in instances_list[j]:
        #                 old_mask = labels[shot[0]][labels[shot[0]] == shot[1]]
        #                 intersection = float(np.count_nonzero(np.logical_and(new_mask, old_mask)))
        #                 iou = intersection / (np.sum(new_mask) + np.sum(old_mask) - intersection)   
        #                 if iou > 0.3:
        #                     iou_dict[j] = iou
                            
        #         if len(iou_dict != 0):
        #             id_list = sorted(iou_dict.items(), key=lambda item:item[1], reverse=True)          
        #             instances_list[id_list[0][0]].append([i,seg_id,compute_bbox(scene,labels[i] == seg_id)])             
        #         else:
        #             instances_list.append([[i,seg_id,compute_bbox(scene,labels[i] == seg_id)]])

        # print(time.time() - time1)
        
        # """========================== end =========================="""
        
        # if not os.path.exists("/remote-home/yangbin/CutLER/maskcut/Ncut_bbox/iter0/"):
        #     os.makedirs("/remote-home/yangbin/CutLER/maskcut/Ncut_bbox/iter0/")
            
        # _save_yaml(f"/remote-home/yangbin/CutLER/maskcut/Ncut_bbox/iter0/{scene_id}.yaml",instances_list)

        # """========================== find bbox =========================="""
        # # bboxs_list = [] # [image_index,object_index]
        # # for j in range(len(instances_list)):  
        # #     # sort instance_list[j] by bbox size
                        
        # #     id_list = sorted(instances_list[j], key=lambda item:(item[2][0] - item[2][1]), reverse=False) 
        
        # """========================== end =========================="""
        
        """========================== merge mask =========================="""
        while len(labels) != 1:
            # print(len(labels), flush=True)
            new_labels = []
            # new_feats = []
            for indice in pairwise_indices(len(labels)):
                # print(indice)
                  
                label = cal_group(labels, indice,scene)
                # label = num_to_natural(label)
                if label is not None:
                    new_labels.append(label)
                    # new_feats.append(feat)
            labels = new_labels
            # feats = new_feats
        # print(labels[0].shape)
        # label = find_pure_seg(labels[0],scene)
        label = labels[0]
        # torch.save(remove_small_group(labels[0], 50), f"proj5/{scene_id}.pth")
        """========================== end =========================="""
        
            
        
        if not os.path.exists("/remote-home/yangbin/CutLER/maskcut/Ncut/iter0/"):
            os.makedirs("/remote-home/yangbin/CutLER/maskcut/Ncut/iter0/")
        torch.save(label, f"/remote-home/yangbin/CutLER/maskcut/Ncut/iter0/{scene_id}.pth")
        # output
        # visualize(scene, labels.long().squeeze(1).cpu().numpy())

    