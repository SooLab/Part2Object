import os 
import queue
from tqdm import tqdm
import torch
import numpy as np
from queue import Queue
import argparse
from project import *
def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]


def calculate_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum(-1)
    union = mask2.sum(-1) + 1e-6

    iou = intersection / union
    return iou

def to_tensor(arr):
    return torch.Tensor(arr).cuda()

def cal_group(label_list, index, ratio=0.5):
    if len(index) == 1:
        return label_list[index[0]]

    group_0 = label_list[index[0]]
    group_1 = label_list[index[1]]

    group_1[group_1 != 0] += group_0.max() + 1
    
    unique_groups_1, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups_1, group_0_counts))
    unique_groups_0, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups_0, group_1_counts))
    
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
        
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        if count / total_count >= ratio:
            
            group_1[group_1 == group_i] = group_j
            
    return group_1

def mask2mask(similarities, mask):
    
    #### fixed patch size
    num_patches1 = [55, 73]
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=similarities.device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0], nn_1[0]
    sim_2, nn_2 = sim_2[0], nn_2[0]
    bbs_mask = nn_2[nn_1] == image_idxs
    if len(mask.shape) == 3:
        mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(num_patches1[0], num_patches1[1]), mode='nearest').view(-1)
    else:
        mask_resized = mask
    new_mask_resized = mask_resized != 0
    new_mask_resized = new_mask_resized.to(similarities.device)
    #### img1's mask
    frame1_mask = torch.bitwise_and(bbs_mask, new_mask_resized.bool())
    
    indices_to_show = torch.nonzero(frame1_mask, as_tuple=False).squeeze(dim=1)  # close bbs
    
    img2_indices_to_show = nn_1[indices_to_show]
    
    img2_mask = torch.zeros(num_patches1[0] * num_patches1[1], device=similarities.device)
    #### img2's mask
    img2_mask[img2_indices_to_show] = 1
    return img2_mask

def full_the_queue(queue: Queue, max_size: int = 5, item=None):
    """ Pops the oldest item from the queue and pushes the new item. """
    if queue.qsize() >= max_size:
        queue.get()
    queue.put(item)

def one2three_check(one_mask, three_masks):
    ### one mask [4015]
    ### three masks [1, 240, 320]
    num_patches1 = [55, 73]
    
    three_mask_resized = torch.nn.functional.interpolate(three_masks.unsqueeze(1), size=(num_patches1[0], num_patches1[1]), mode='nearest').view(-1)
    three_mask_resized = three_mask_resized.to(one_mask.device)
    three_mask_resized = three_mask_resized.unsqueeze(0).repeat(3, 1)
    
    three_mask_resized = three_mask_resized == torch.arange(1, 4, device=three_mask_resized.device).unsqueeze(1)
    
    return calculate_iou(one_mask.bool(), three_mask_resized)

def merge_no_draw(labels):

    while len(labels) != 1:
        # print(len(labels), flush=True)
        new_labels = []
        # new_feats = []
        for indice in pairwise_indices(len(labels)):
            label = cal_group(labels, indice)
            # label = num_to_natural(label)
            if label is not None:
                new_labels.append(label)
                # new_feats.append(feat)
        labels = new_labels

    return labels[0]


# ============================= new add ========================================
thre = 0.3
# os.makedirs(f"shicheng_data_ab/_t{thre}", exist_ok=True)
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Part2Object script')
    # default arguments
    parser.add_argument('--immedia_data_dir', type=str, default="data_immedia_result", help='matrix and label stroage')
    parser.add_argument('--scannet_frames', type=str, default="Scannetv2/frames_square/", help='frames')
    parser.add_argument('--val_data_dir', type=str, default="SegmentAnything3D/data/val", help='val datasets')
    parser.add_argument('--output_dir', type=str, default="part2object_data/", help='output dirs')
    args = parser.parse_args()
        
  
    # ============================ new add =========================================
    files = sorted(os.listdir(args.val_data_dir)) 
    files = [f.replace(".pth","") for f in files if f.endswith(".pth")]
    
    # =============================================================================
    
    for scene_id in tqdm(files):
        
        if os.path.exists(f"{args.output_dir}/{scene_id}_merge_ncut.pth".format(scene_id)):
            continue
        
        output_label = torch.load(f"{args.immedia_data_dir}/{scene_id}_label.pth".format(scene_id), map_location=torch.device('cpu'))
        scene_matrix = torch.load(f"{args.immedia_data_dir}/{scene_id}_matrix.pth".format(scene_id), map_location=torch.device('cpu'))
        pngs = sorted(os.listdir('outputs/'+scene_id), key=lambda x: int(x.split(".")[0]))
        mask_queue = queue.Queue()
        top_id = 0
        end_id = 0
        silde_window = 3
        
        stack_output_label = np.stack(output_label)

        while top_id < len(pngs):
            while end_id - top_id < silde_window and end_id < len(pngs):
                mask_path = 'outputs/'+scene_id+'/'+pngs[end_id]
                feat = to_tensor(load_image(mask_path, [320, 240])).unsqueeze(0)
                mask_queue.put(feat)
                end_id += 1
                
            input_mask = mask_queue.get()
            
            mask_num = int(input_mask.max())
            
            for i in range(mask_num):
                mask_in = (input_mask == i+1).float()
                
                for j in range(mask_queue.qsize()):
                    mask_in = mask2mask(scene_matrix[top_id+j], mask_in)
                    iou = one2three_check(mask_in, mask_queue.queue[j]) > thre
                    if iou.sum() > 1:
                        iou = torch.max(one2three_check(mask_in, mask_queue.queue[j]),-1)[1]
                        try:
                            mask_3d_1 = output_label[top_id] == i+1
                            mask_3d_2 = output_label[top_id+j+1] == iou.item() + 1
                            merge_mask = np.bitwise_or(mask_3d_1, mask_3d_2)
                            output_label[top_id][merge_mask] = i+1
                            output_label[top_id+j+1][merge_mask] = iou.item() + 1
                        except IndexError:
                            continue
                    else:
                        iou = torch.nonzero(iou, as_tuple=False)
                        if iou.shape[0] != 0:
                            try:
                                mask_3d_1 = output_label[top_id] == i+1
                                mask_3d_2 = output_label[top_id+j+1] == iou.item() + 1
                                merge_mask = np.bitwise_or(mask_3d_1, mask_3d_2)
                                output_label[top_id][merge_mask] = i+1
                                output_label[top_id+j+1][merge_mask] = iou.item() + 1
                            except IndexError:
                                continue
            top_id += 1
            # break
        out = merge_no_draw(output_label)
        torch.save(out, f"{args.output_dir}/{scene_id}_merge_ncut.pth".format(scene_id))
