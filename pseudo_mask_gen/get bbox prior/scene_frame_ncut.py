import torch
import numpy as np
from PIL import Image
import math
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os 
from correspondences import *
from sklearn.cluster import DBSCAN
import argparse
from project import *
import open3d as o3d

def color_generate(input_matrix:np.ndarray):
    # input : N * 1 label
    # output : n*3 color
    color_dict = {0:[1,1,1]}
    output_matrix = np.zeros((input_matrix.shape[0], 3))
    for i in range(input_matrix.shape[0]):
        label = input_matrix[i]
        if label in color_dict:
            color = color_dict[label]
            output_matrix[i, :] = color
        else:
            color = np.random.rand(3)
            color_dict[label] = color
            output_matrix[i, :] = color

    return output_matrix

def project(frame_id, scene_id):
    
    # draw
    try:
        scene = torch.load(os.path.join(SCANNET_DATA, "train", scene_id)+".pth")["coord"]
    except:
        scene = torch.load(os.path.join(SCANNET_DATA, "val", scene_id)+".pth")["coord"]
        
    frame_list = [frame_id]
    # scene_images = np.zeros((len(frame_list), 3, 256, 328))
    scene_depths = np.zeros((len(frame_list), 240, 320))
    scene_poses = np.zeros((len(frame_list), 4, 4))
    for i, frame_id in enumerate(frame_list):
        # scene_images[i] = load_image(SCANNET_FRAME_PATH.format(scene_id) + "/color" + "/{}.jpg".format(frame_id), [328, 256])
        scene_depths[i] = load_depth(SCANNET_FRAME_PATH.format(scene_id) + "/depth" + "/{}.png".format(frame_id), [320, 240])
        scene_poses[i] = load_pose(SCANNET_FRAME_PATH.format(scene_id) + "/pose" + "/{}.txt".format(frame_id))
        
    projection_3d, projection_2d = compute_projection(scene, scene_depths, scene_poses)
        
    feat = to_tensor(load_image(f"outputs/{scene_id}/" + "/{}.png".format(frame_id), [320, 240])).unsqueeze(0)
    proj_label = PROJECTOR.project(feat, projection_3d[0], projection_2d[0], scene.shape[0]).transpose(1, 0)
    proj_label = proj_label.int().cpu().numpy().reshape(proj_label.shape[0],)
    return proj_label, scene

def to_tensor(arr):
    return torch.Tensor(arr)

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    
    return x @ y.transpose(-1, -2)

def find_corresp_by_mask(image_path1, image_path2, extractor, num_pairs = 100, load_size = 224, mask = None, mask2 = None,device = 'cuda'):
    layer = 9
    facet = 'key'
    thresh = 0.05
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)
    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    return similarities

def find_connected_components(point_cloud, eps=0.1, min_samples=6):
    # Filter the point cloud using the mask
    filtered_point_cloud = point_cloud

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_point_cloud)
    labels = clustering.labels_

    # Create an array for the labels of all points
    all_labels = np.zeros_like(point_cloud, dtype=int)  # Initialize with -1

    # Assign the labels to the points where mask == 1
    all_labels = (labels + 1)

    return all_labels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Part2obj script')
    # default arguments
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id')
    parser.add_argument('--train_data_dir', type=str, default="SegmentAnything3D/data/train", help='')
    parser.add_argument('--immedia_data_dir', type=str, default="data_immedia_result", help='matrix and label stroage')
    parser.add_argument('--scannet_frames', type=str, default="Scannetv2/frames_square/", help='frames')
    args = parser.parse_args()

    # ============================ new add =========================================
    files = sorted(os.listdir(args.train_data_dir)) 
    files = [f.replace(".pth","") for f in files if f.endswith(".pth")]
    # =============================================================================
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = f'cuda:{args.gpu}'
    extractor = ViTExtractor('dino_vits8', 4, device=device)
    os.makedirs(args.immedia_data_dir, exist_ok=True)
    
    
    for scene_idx in tqdm(range(len(files))):

        scene_id = files[scene_idx]
        if os.path.exists(f"{args.immedia_data_dir}/{scene_id}_matrix.pth") and os.path.exists(f"{args.immedia_data_dir}/{scene_id}_label.pth"):
            continue
        
        pngs = sorted(os.listdir('outputs/'+scene_id), key=lambda x: int(x.split(".")[0]))

        object_memory = []
        output_label = []
        scene_matrix = []
        for frame_id, png in enumerate(pngs):

            mask_path = 'outputs/'+scene_id+'/'+pngs[frame_id]
            image = plt.imread(mask_path)
            np_image = np.asarray(image)
            try:
                mask_path2 = 'outputs/'+scene_id+'/'+pngs[frame_id+1]
            except IndexError:
                break
            image_path1 = args.scannet_frames + scene_id + "/" + 'color/' + pngs[frame_id].replace('png', 'jpg')
            image_path2 = args.scannet_frames + scene_id + "/" + 'color/' + pngs[frame_id+1].replace('png', 'jpg')
            feat = to_tensor(load_image(mask_path, [320, 240])).unsqueeze(0)
            feat2 = to_tensor(load_image(mask_path2, [320, 240])).unsqueeze(0) 

            with torch.no_grad():
                matrix = find_corresp_by_mask(image_path1, image_path2, extractor, num_pairs=20, mask = feat, mask2 =feat2,device=device)
                scene_matrix.append(matrix.squeeze(0))

            proj_label, scene = project(pngs[frame_id].split('.')[0], scene_id)
            for i in range(int(np_image.max()*255)):
                object1_mask = scene[proj_label == i+1]
                if object1_mask.shape[0] ==0:
                    continue
                
                
                # clean by distance
                label = find_connected_components(object1_mask,  eps=0.1)
                if label.max() in [2,3,4]:
                    counts = np.bincount(label)
                    select = np.argmax(counts)
                    temp = proj_label[proj_label == i+1]
                    temp[label != select] = 0
                    proj_label[proj_label == i+1] = temp
                elif label.max() > 4:
                    proj_label[proj_label == i+1] = 0
            output_label.append(proj_label)

        torch.save(scene_matrix, f"{args.immedia_data_dir}/{scene_id}_matrix.pth".format(scene_id))
        torch.save(output_label, f"{args.immedia_data_dir}/{scene_id}_label.pth".format(scene_id))