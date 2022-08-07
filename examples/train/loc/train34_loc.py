import os
from os import path, makedirs, listdir
import sys
import numpy as np

np.random.seed(1)
import random

random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from src.optim.adamw import AdamW
from src.losses import dice_round, ComboLoss

from tqdm import tqdm
import timeit
import cv2

from src.zoo.models import Res34_Unet_Loc

from imgaug import augmenters as iaa

from src.util.augmentations import (
    shift_image,
    rotate_image,
    shift_channels,
    change_hsv,
    clahe,
    gauss_noise,
    blur,
    saturation,
    brightness,
    contrast
)
from src.util.utils import normalize_colors

from sklearn.model_selection import train_test_split

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

input_shape = (736, 736)


def validate(net, data_loader):
    dices0 = []

    _thr = 0.5

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)

            out = model(imgs)

            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()

            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

    d0 = np.mean(dices0)

    print("Val Dice: {}".format(d0))
    return d0


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch(current_epoch, seg_loss, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)

        out = model(imgs)

        loss = seg_loss(out, msks)

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(loss.item(), imgs.size(0))

        dices.update(dice_sc, imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Dice {dice.avg:.4f}".format(
        current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)

    seed = int(sys.argv[1])
    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    cudnn.benchmark = True



    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
