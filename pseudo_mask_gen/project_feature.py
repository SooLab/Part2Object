import warnings

warnings.filterwarnings("ignore")

import os
# import sys
# import h5py
import torch
# import torch.nn as nn
# import argparse
import numpy as np
from tqdm import tqdm
# from plyfile import PlyData, PlyElement
import math
from imageio import imread
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import zoom
import cv2

# sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
# from lib.config import CONF
from projection import ProjectionHelper
import copy

# SCANNET_LIST = CONF.SCANNETV2_LIST
# SCANNET_DATA = CONF.PATH.SCANNET_DATA
# SCANNET_FRAME_ROOT = CONF.SCANNET_FRAMES
# SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, "{}") # name of the file

SCANNET_DATA = 'data'
SCANNET_FRAME_ROOT = 'data/frames_square'
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, "{}") # name of the file


# ENET_FEATURE_PATH = CONF.ENET_FEATURES_PATH
# ENET_FEATURE_DATABASE = CONF.MULTIVIEW

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

# projection
INTRINSICS = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
INTRINSICS = adjust_intrinsic(INTRINSICS, (41, 32), (320, 240))

PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [320, 240], 0.05)


def to_tensor(arr):
    return torch.Tensor(arr).cuda()

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

def get_scene_data(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        # load the original vertices, not the axis-aligned ones
        scene_data[scene_id] = np.load(os.path.join(SCANNET_DATA, scene_id)+"_vert.npy")[:, :3]
    
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
    indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(to_tensor(points), to_tensor(depth[i]), to_tensor(camera_to_world[i]))
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()
            # print("found {} mappings in {} points from frame {}".format(indices_3ds[i][0], num_points, i))
        
    return indices_3ds, indices_2ds

if __name__ == "__main__":
    # args = CONF

    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    maxpool = True
    feat_dim = 768
    scale_factor = 4


    # scene_list = get_scene_list()
    scene_list = sorted(os.listdir("data/frames_square/"))
    dinov2_vitb14 = torch.hub.load('dinov2', 'dinov2_vitb14', source='local').cuda().eval()

    print("projecting multiview features to point cloud...")
    for scene_id in scene_list:
        print("processing {}...".format(scene_id))

        try:
            scene = torch.load(os.path.join(SCANNET_DATA, "train", scene_id)+".pth")["coord"]
        except:
            scene = torch.load(os.path.join(SCANNET_DATA, "val", scene_id)+".pth")["coord"]

        # load frames
        # frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir(SCANNET_FRAME_ROOT.format(scene_id, "color")))))
        frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir("results8_2d/{}".format(scene_id)))))
        scene_images = np.zeros((len(frame_list), 3, 256, 328))
        scene_depths = np.zeros((len(frame_list), 240, 320))
        scene_poses = np.zeros((len(frame_list), 4, 4))
        for i, frame_id in enumerate(frame_list):
            scene_images[i] = load_image(SCANNET_FRAME_PATH.format(scene_id) + "/color" + "/{}.jpg".format(frame_id), [328, 256])
            scene_depths[i] = load_depth(SCANNET_FRAME_PATH.format(scene_id) + "/depth" + "/{}.png".format(frame_id), [320, 240])
            scene_poses[i] = load_pose(SCANNET_FRAME_PATH.format(scene_id) + "/pose" + "/{}.txt".format(frame_id))

            # color = cv2.imread(SCANNET_FRAME_PATH.format(scene_id) + "/color" + "/{}.jpg".format(frame_id))
            color = Image.open(SCANNET_FRAME_PATH.format(scene_id) + "/color" + "/{}.jpg".format(frame_id))
            color = np.array(color)
            color = zoom(color, (scale_factor, scale_factor, 1), order=3)
            color = resize_crop_image(color, (1274, 952))
            color =  np.transpose(color, [2, 0, 1])
            color = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(color.astype(np.float32) / 255.0))
            color = to_tensor(color).unsqueeze(0)
            
            with torch.no_grad():
                image_embedding = dinov2_vitb14(color).reshape((68, 91, 768)).permute(2, 0, 1).unsqueeze(0)
                image_embedding = torch.nn.functional.interpolate(image_embedding, (240, 320), mode='bicubic')
            # print(image_embedding.shape)
            # break
            
            torch.save(image_embedding.cpu().detach().numpy(), f"DINO_feats/{frame_id}.pth")


        # compute projections for each chunk
        projection_3d, projection_2d = compute_projection(scene, scene_depths, scene_poses)
        
        # compute valid projections
        projections = []
        for i in range(projection_3d.shape[0]):
            num_valid = projection_3d[i, 0]
            if num_valid == 0:
                continue

            projections.append((frame_list[i], projection_3d[i], projection_2d[i]))


        # project
        point_features = to_tensor(scene).new(scene.shape[0], feat_dim).fill_(0)
        num_feats = torch.ones(scene.shape[0], feat_dim).cuda()
        for i, projection in enumerate(projections):
            frame_id = projection[0]
            projection_3d = projection[1]
            projection_2d = projection[2]
            
            feat = to_tensor(torch.load(f"DINO_feats/{frame_id}.pth"))
            feat = feat.squeeze(0)
            
            proj_feat = PROJECTOR.project(feat, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0)
            

            if maxpool:
                # only apply max pooling on the overlapping points
                # find out the points that are covered in projection
                feat_mask = ((proj_feat == 0).sum(1) != feat_dim).bool()
                # find out the points that are not filled with features
                point_mask = ((point_features == 0).sum(1) == feat_dim).bool()

                # for the points that are not filled with features
                # and are covered in projection, 
                # simply fill those points with projected features
                mask = point_mask * feat_mask
                point_features[mask] = proj_feat[mask]

                # for the points that have already been filled with features
                # and are covered in projection, 
                # apply max pooling first and then fill with pooled values
                mask = ~point_mask * feat_mask
                if mask.sum() > 0:
                    norm_1 = point_features[mask] / point_features[mask].norm(dim=0, keepdim=True)
                    norm_2 = proj_feat[mask] / proj_feat[mask].norm(dim=0, keepdim=True)
                    sim = norm_1 @ norm_2.T
                    mask_ = torch.diag(sim) >= 0.4
                    mask__ = copy.deepcopy(mask)
                    mask__[mask] = mask_
                    mask = mask * mask__
                # point_features[mask] = torch.max(point_features[mask], proj_feat[mask])
                    point_features[mask] = point_features[mask] + proj_feat[mask]
                    num_feats[mask] += 1

            else:
                if i == 0:
                    point_features = proj_feat
                else:
                    mask = (point_features == 0).sum(1) == feat_dim
                    point_features[mask] = proj_feat[mask]

        # save    
        point_features = point_features / num_feats
        torch.save(point_features.cpu().numpy(), f"DINO_point_feats/{scene_id}.pth")

    print("done!")

    
