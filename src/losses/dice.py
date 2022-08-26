import torch
import numpy as np
import numpy.typing as npt

from .epsilon import eps


def dice(im1: torch.BoolTensor, im2: torch.BoolTensor, empty_score: float = 1.0) -> float:
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : torch.BoolTensor
        Mask 1
    im2 : torch.BoolTensor
        Mask 2
    empty_score: float
        Return Value if Union is empty
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    """

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum: torch.Tensor = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection: torch.Tensor = torch.logical_and(im1, im2)

    return float(2. * intersection.sum() / im_sum)


def dice_batch(msk_batch1: torch.BoolTensor, msk_batch2: torch.BoolTensor,
               empty_score: float = 1.0) -> torch.FloatTensor:
    """
    batch version of dice function
    :param msk_batch1: torch.BoolTensor [B,H,W]
    :param msk_batch2: torch.BoolTensor [B,H,W]
    :param empty_score: if both masks are empty

    :return dices: torch.FloatTensor [W]
    """

    if msk_batch1.shape != msk_batch2.shape:
        raise ValueError("Shape mismatch: msk_batch1 and msk_batch2 must have the same shape.")

    msk_sum_batch: torch.Tensor = msk_batch1.sum(dim=(1, 2)) + msk_batch2.sum(dim=(1, 2))

    intersection_batch: torch.Tensor = torch.logical_and(msk_batch1, msk_batch2).sum(dim=(1, 2))

    dices = 2 * intersection_batch / msk_sum_batch

    dices[dices.isinf() | dices.isnan()] = empty_score

    return dices


def soft_dice_loss(outputs: torch.Tensor, targets: torch.Tensor, per_image=False) -> torch.Tensor:
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def dice_round(preds, trues) -> torch.Tensor:
    preds = preds.float()
    return soft_dice_loss(preds, trues)


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image)
