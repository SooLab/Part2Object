# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1)


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score  

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 class_weights):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels" + mask_type[-7:]][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=253)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def mask_included(self, mask1, mask2, coords):
        containment = torch.all(mask1 <= mask2).item()
        min_1 = coords[mask1].min(dim=0).values
        max_1 = coords[mask1].max(dim=0).values

        area1 = torch.prod((max_1 - min_1) ** 2)

        min_2 = coords[mask2].min(dim=0).values
        max_2 = coords[mask2].max(dim=0).values

        area2 = torch.prod((max_2 - min_2) ** 2)

        
        return containment, area2 / area1
    
    def computer_box_loss(self, mask1, mask2, correction_factor, coords):
        if mask2.sum() == 0 or mask1.sum() == 0:
            return 1
        min_1 = coords[mask1].min(dim=0).values
        max_1 = coords[mask1].max(dim=0).values

        # print(mask2)
        min_2 = coords[mask2].min(dim=0).values
        max_2 = coords[mask2].max(dim=0).values

        intersection_volume = torch.clamp(torch.min(max_1, max_2) - torch.max(min_1, min_2), min=0).prod()
        union_volume = (max_1 - min_1).prod() + (max_2 - min_2).prod() - intersection_volume

        giou = intersection_volume / union_volume * correction_factor

        giou_loss = 1 - giou

        return giou_loss

    
    def box_loss(self, part_indice, obj_indice, part_target, obj_target, pred_part, pred_obj, mask_type_part, mask_type_obj, data):
        loss_boxes = []
        for batch_id, (map_id, target_id) in enumerate(part_indice):
            box_loss = torch.tensor(0.0).cuda()
            num_need_box = 1e-6

            coords = data.decomposed_coordinates[batch_id] / (data.decomposed_coordinates[batch_id].float().norm(dim=1, keepdim=True) + 1e-6)
            try:
                coords = scatter_mean(coords, part_target[batch_id]["point2segment"], dim=0)
            except:
                print(len(part_target))
                print(part_target[batch_id]["point2segment"].shape)
                print(coords.shape)
                exit()
            # print(coords.shape)
            # exit()
            map_part = pred_part["pred_masks"][batch_id][:, map_id].T
            target_mask_part = part_target[batch_id][mask_type_part][target_id]

            (map_id_o, target_id_o) = obj_indice[batch_id]
            map_obj = pred_obj["pred_masks"][batch_id][:, map_id_o].T
            target_mask_obj = obj_target[batch_id][mask_type_obj][target_id_o]

            for i, part_target_mask in enumerate(target_mask_part):
                for j, obj_target_mask in enumerate(target_mask_obj):
                    # con, ratio = self.mask_included(part_target_mask, obj_target_mask, coords)
                    try:
                        con = part_target[batch_id]["include_matric"][i, j]
                        ratio = part_target[batch_id]["ratio_matric"][i, j]
                    except:
                        pass
                        
                    if con:
                        box_loss += self.computer_box_loss(map_part[i] > 0, map_obj[j] > 0, ratio, coords)
                        num_need_box += 1
            loss_boxes.append(box_loss / num_need_box)

        return torch.sum(torch.stack(loss_boxes))



    def loss_masks(self, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []
        loss_scores = []

        drop_loss_th = 0.01

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]

            pred_mask = map > 0
            union = (pred_mask | target_mask)
            intersect = (pred_mask & target_mask)

            IoU = intersect.sum(dim=1) / union.sum(dim=1)
            # print(IoU)
            drop_loss_weight = IoU.le(drop_loss_th).float()
            drop_loss_weight = 1 - drop_loss_weight.ge(1.0).float()

            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1],
                                           device=target_mask.device)[:int(self.num_points*target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()
            pred_score = outputs['pred_scores'][batch_id]
            target_scores = torch.zeros_like(pred_score)
            target_scores[map_id] = 1
            loss_scores.append(self.sigmoid_focal_loss(pred_score, target_scores))
            loss_masks_sub = sigmoid_ce_loss_jit(map,
                                                  target_mask,
                                                  num_masks)
            
            loss_dices_sub = dice_loss_jit(map,
                                            target_mask,
                                            num_masks)
            
            loss_masks.append(loss_masks_sub.sum() / num_masks)
            loss_dices.append(loss_dices_sub.sum() / num_masks)
            
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
            "loss_score": torch.sum(torch.stack(loss_scores))
        }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t[mask_type] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, mask_type),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks, mask_type),
        }

        del src_masks
        del target_masks
        return losses
    
    def score_loss(self, indices, scores):

        # num_topk = [200, 150, 120, 100, 80]
        batch_size = len(indices)

        target = torch.zeros_like(scores)
        for b in range(batch_size):
            target[b, indices[b][0]] = 1


        score_loss = self.sigmoid_focal_loss(scores, target)

        return {f'score_loss': score_loss}


    def sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """

        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type)

    def forward(self, outputs, targets, mask_type, epoch, data):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # layer = epoch // 120 + 1
        layer = 2
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # all_indices = {}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_1 = self.matcher(outputs_without_aux, targets, f"segment_mask_layer{layer}")

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t[f"labels_layer{layer}"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()


        num_masks_part = sum(len(t[f"labels_layer1"]) for t in targets)
        num_masks_part = torch.as_tensor(
            [num_masks_part], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_part)
        num_masks_part = torch.clamp(num_masks_part / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices_1, num_masks, f"segment_mask_layer{layer}"))


        last_layer = 0
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                layer = 2
                indices = self.matcher(aux_outputs, targets, f"segment_mask_layer{layer}", score_outputs=outputs)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, f"segment_mask_layer{layer}")
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        layer = 1
        indices_2 = self.matcher(outputs['pred_part'], targets, f"segment_mask_layer{layer}")
        # Compute part loss
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs['pred_part'], targets, indices_2, num_masks_part, f"segment_mask_layer{layer}")
            l_dict = {k + f"_part": v for k, v in l_dict.items()}
            losses.update(l_dict)

        for i, aux_outputs in enumerate(outputs["part_results"]):
            indices = self.matcher(aux_outputs, targets, f"segment_mask_layer{layer}", score_outputs=outputs)
            for loss in self.losses:
                l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks_part, f"segment_mask_layer{layer}")
                l_dict = {k + f"_{i}_part": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, indices_1

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
