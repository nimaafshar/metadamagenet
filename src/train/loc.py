import abc
from typing import Union, Optional
import dataclasses
from contextlib import nullcontext

import numpy as np
from torch import nn
import torch
from torch.cuda import amp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.backends import cudnn
from tqdm import tqdm

from .trainer import Trainer, TrainingConfig
from src.util.utils import AverageMeter
from src.losses import dice_batch, dice_round, ComboLoss
from src.logs import log


@dataclasses.dataclass
class LocalizationRequirements:
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: MultiStepLR
    seg_loss: ComboLoss
    grad_scaler: Optional[amp.GradScaler] = None


class LocalizationTrainer(Trainer, abc.ABC):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        requirements: LocalizationRequirements = self._get_requirements()
        self._model: nn.Module = requirements.model
        self._optimizer: Optimizer = requirements.optimizer
        self._lr_scheduler: MultiStepLR = requirements.lr_scheduler
        self._seg_loss: ComboLoss = requirements.seg_loss
        self._grad_scaler: Optional[amp.GradScaler] = requirements.grad_scaler
        self._evaluation_dice_thr: float = 0.5

    @abc.abstractmethod
    def _get_requirements(self) -> LocalizationRequirements:
        pass

    @abc.abstractmethod
    def _update_weights(self, loss: torch.Tensor) -> None:
        pass

    def _setup(self):
        super(LocalizationTrainer, self)._setup()
        # vis_dev = sys.argv[2]
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev
        cudnn.benchmark = True

    def _evaluate(self, number: int) -> float:

        meter = AverageMeter()

        iterator = tqdm(self._val_data_loader)
        self._model.eval()
        with torch.no_grad():
            for i, (img_batch, msk_batch) in enumerate(iterator):
                msk_batch: torch.FloatTensor = msk_batch.cuda(non_blocking=True)
                img_batch: torch.BoolTensor = img_batch.cuda(non_blocking=True)

                out_batch: torch.FloatTensor = self._model(img_batch)
                msk_pred: torch.FloatTensor = torch.sigmoid(out_batch[:, 0, ...])

                dice_scores = dice_batch(msk_batch[:, 0, ...], msk_pred > self._evaluation_dice_thr)
                meter.update(float(dice_scores.mean()), n=img_batch.size(0))

                iterator.set_description(
                    f"epoch: {number};'"
                    f" Dice {meter.val:.6f} ({meter.avg:.6f})")

        log(f"Validation set Dice: {meter.avg:.6f}")
        return meter.avg

    def _save_model(self, epoch: int, score: float, best_score: Union[float, None]) -> bool:
        if best_score is None or score > best_score:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'best_score': score,
            }, self._config.model_config.best_snap_path)
            log(f":floppy_disk: model saved at {self._config.model_config.best_snap_path}")
            return True

        return False

    def _update_best_score(self, score: float, best_score: Union[float, None]) -> float:
        if best_score is None:
            log(f"score={score:.4f}")
            return score

        if score > best_score:
            log(f":confetti_ball: score {best_score:.4f} --> {score:.4f}")
            return score
        else:
            log(f":disappointed: score {best_score:.4f} --> {score:.4f}")
            return best_score

    def _train_epoch(self, epoch: int):
        losses_meter: AverageMeter = AverageMeter()
        dices_meter: AverageMeter = AverageMeter()

        self._model.train()

        iterator = tqdm(self._train_data_loader)

        for i, (img_batch, msk_batch) in enumerate(iterator):
            img_batch: torch.Tensor = img_batch.cuda(non_blocking=True)
            msk_batch: torch.Tensor = msk_batch.cuda(non_blocking=True)

            with amp.autocast() if self._grad_scaler is not None else nullcontext():
                out: torch.Tensor = self._model(img_batch)
                loss: torch.Tensor = self._seg_loss(out, msk_batch)

            with torch.no_grad():
                _probs = torch.sigmoid(out[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, msk_batch[:, 0, ...])

            losses_meter.update(loss.item(), img_batch.size(0))

            dices_meter.update(dice_sc, img_batch.size(0))

            iterator.set_description(
                f"epoch: {epoch};'"
                f" lr {self._lr_scheduler.get_last_lr()[-1]:.7f};"
                f" Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f});"
                f" Dice {dices_meter.val:.4f} ({dices_meter.avg:.4f})")

            self._update_weights(loss)

        self._lr_scheduler.step()

        log(f"epoch: {epoch};"
            f"lr {self._lr_scheduler.get_last_lr()[-1]:.7f};"
            f"Loss {losses_meter.avg:.4f};"
            f"Dice {dices_meter.avg:.4f}")
