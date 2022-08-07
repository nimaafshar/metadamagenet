import abc
from typing import Union
import dataclasses

import numpy as np
from torch import nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.backends import cudnn
from tqdm import tqdm

from .trainer import Trainer, TrainingConfig
from src.util.utils import AverageMeter, dice
from src.losses import dice_round, ComboLoss
from src.logs import log


@dataclasses.dataclass
class LocalizationRequirements:
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: MultiStepLR
    seg_loss: ComboLoss


class LocalizationTrainer(Trainer):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        requirements: LocalizationRequirements = self._get_requirements()
        self._model: nn.Module = requirements.model
        self._optimizer: Optimizer = requirements.optimizer
        self._lr_scheduler: MultiStepLR = requirements.lr_scheduler
        self._seg_loss: ComboLoss = requirements.seg_loss
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

        dices0 = []

        self._model.eval()
        with torch.no_grad():
            for i, (img_batch, msk_batch) in enumerate(tqdm(self._val_data_loader)):
                msk_batch = msk_batch.numpy()
                img_batch = img_batch.cuda(non_blocking=True)

                out_batch = self._model(img_batch)

                msk_pred = torch.sigmoid(out_batch[:, 0, ...]).cpu().numpy()

                for j in range(msk_batch.shape[0]):
                    dices0.append(dice(msk_batch[j, 0], msk_pred[j] > self._evaluation_dice_thr))

        d0: float = np.mean(dices0)

        log(f"Validation set Dice: {d0:.6f}")
        return d0

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

            out: torch.Tensor = self._model(img_batch)

            loss: torch.Tensor = self._seg_loss(out, msk_batch)

            with torch.no_grad():
                _probs = torch.sigmoid(out[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, msk_batch[:, 0, ...])

            losses_meter.update(loss.item(), img_batch.size(0))

            dices_meter.update(dice_sc, img_batch.size(0))

            # TODO: test get_lr() method
            iterator.set_description(
                f"epoch: {epoch};'"
                f" lr {self._lr_scheduler.get_lr()[-1]:.7f};"
                f" Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f});"
                f" Dice {dices_meter.val:.4f} ({dices_meter.avg:.4f})")

            self._update_weights(loss)

        self._lr_scheduler.step(epoch)

        log(f"epoch: {epoch}; lr {self._lr_scheduler.get_lr()[-1]:.7f}; Loss {losses_meter.avg:.4f}; Dice {dices_meter.avg:.4f}")


