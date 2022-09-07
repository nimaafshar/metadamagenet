import torch
import numpy as np

from .epsilon import eps


def iou(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    union = np.logical_or(im1, im2)
    im_sum = union.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum


def jaccard(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) - intersection + eps
    losses = 1 - (intersection + eps) / union
    return losses.mean()


def iou_round(preds, trues):
    preds = preds.float()
    return jaccard(preds, trues)


class JaccardLoss(torch.nn.Module):
    """
    expects sigmoid
    """
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, outputs, targets):
        return jaccard(outputs, targets, per_image=self.per_image)
