import abc
from typing import Union, List, Optional
import dataclasses
from contextlib import nullcontext

import torch
import numpy as np
import numpy.typing as npt
from torch.backends import cudnn
from torch import nn
from torch.cuda import amp
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from src.util.utils import AverageMeter
from .trainer import Trainer, TrainingConfig, Requirements
from src.losses import ComboLoss, dice_round
from src.logs import log
from .metrics import DiceCalculator, MetricCalculator


class ClassificationRequirements(Requirements):
    ce_loss: nn.CrossEntropyLoss
    label_loss_weights: npt.NDArray  # with size 5, if using cce_loss use size 6
    dice_metric_calculator: Optional[MetricCalculator] = None


class ClassificationTrainer(Trainer, abc.ABC):

    def _setup(self):
        super(ClassificationTrainer, self)._setup()
        # vis_dev = sys.argv[2]
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev
        cudnn.benchmark = True

    def __init__(self, config: TrainingConfig,
                 use_cce_loss: bool = False):
        requirements: ClassificationRequirements = self._get_requirements()
        super().__init__(config, requirements)
        self._ce_loss: nn.CrossEntropyLoss = requirements.ce_loss
        self._label_loss_weights: torch.Tensor = torch.from_numpy(requirements.label_loss_weights)
        self._evaluation_dice_thr: float = 0.3
        self._use_cce_loss: bool = use_cce_loss
        self._dice_metric_calculator: MetricCalculator = requirements.dice_metric_calculator if \
            requirements.dice_metric_calculator is not None else DiceCalculator(self._evaluation_dice_thr)

    @abc.abstractmethod
    def _get_requirements(self) -> ClassificationRequirements:
        pass

    @abc.abstractmethod
    def _update_weights(self, loss: torch.Tensor) -> None:
        pass

    @abc.abstractmethod
    def _apply_activation(self, model_out: torch.Tensor) -> torch.Tensor:
        """
        :param model_out: batch of model outputs
        :return: model outputs with activation function applied
        """
        pass

    def _evaluate(self, number: int) -> float:
        self._dice_metric_calculator.reset()

        tp = np.zeros((4,))
        fp = np.zeros((4,))
        fn = np.zeros((4,))

        self._model.eval()
        with torch.no_grad():
            for i, valid_data_batch in enumerate(tqdm(self._val_data_loader)):
                msk_batch: torch.Tensor = valid_data_batch['msk'].numpy()
                label_mask_batch: torch.Tensor = valid_data_batch['label_msk'].numpy()
                img_batch: torch.Tensor = valid_data_batch['img'].cuda(non_blocking=True)
                loc_mask_batch: torch.Tensor = valid_data_batch['msk_loc'].numpy() * 1

                out_batch = self._model(img_batch)

                msk_loc_pred = loc_mask_batch
                msk_damage_pred = self._apply_activation(out_batch).cpu().numpy()[:, 1:, ...]

                for j in range(msk_batch.shape[0]):
                    self._dice_metric_calculator.update(msk_batch[j, 0], msk_loc_pred[j])

                    targ = label_mask_batch[j][msk_batch[j, 0] > 0]
                    pred = msk_damage_pred[j].argmax(axis=0)
                    pred = pred * (msk_loc_pred[j] > self._evaluation_dice_thr)
                    pred = pred[msk_batch[j, 0] > 0]

                    for c in range(4):
                        tp[c] += np.logical_and(pred == c, targ == c).sum()
                        fn[c] += np.logical_and(pred != c, targ == c).sum()
                        fp[c] += np.logical_and(pred == c, targ != c).sum()

        d0: float = self._dice_metric_calculator.aggregate()

        f1_scores = np.zeros((4,))
        for c in range(4):
            f1_scores[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])

        f1_final = 4 / np.sum(1.0 / (f1_scores + 1e-6))

        score = 0.3 * d0 + 0.7 * f1_final

        log(f"Validation set Score: "
            f"{score:.6f}, "
            f"Dice: {d0:.6f}, "
            f"F1: {f1_final}, "
            f"F1_0: {f1_scores[0]}, "
            f"F1_1: {f1_scores[1]}, "
            f"F1_2: {f1_scores[2]}, "
            f"F1_3: {f1_scores[3]}")
        return score

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
        label_losses_meter: List[AverageMeter] = [AverageMeter() for _ in range(5)]
        ce_loss_meter: AverageMeter = AverageMeter()

        dices_meter: AverageMeter = AverageMeter()

        self._model.train()

        iterator = tqdm(self._train_data_loader)

        for i, train_data_batch in enumerate(iterator):
            img_batch: torch.Tensor = train_data_batch['img'].cuda(non_blocking=True)
            msk_batch: torch.Tensor = train_data_batch['msk'].cuda(non_blocking=True)
            label_msk_batch: torch.Tensor = train_data_batch['label_msk'].cuda(non_blocking=True)

            with amp.autocast() if self._grad_scaler is not None else nullcontext():
                out: torch.Tensor = self._model(img_batch)

                label_losses: List[torch.Tensor] = [self._seg_loss(out[:, 0, ...], msk_batch[:, 0, ...]),
                                                    self._seg_loss(out[:, 1, ...], msk_batch[:, 1, ...]),
                                                    self._seg_loss(out[:, 2, ...], msk_batch[:, 2, ...]),
                                                    self._seg_loss(out[:, 3, ...], msk_batch[:, 3, ...]),
                                                    self._seg_loss(out[:, 4, ...], msk_batch[:, 4, ...])]
                if self._use_cce_loss:
                    label_losses.append(self._ce_loss(out, label_msk_batch))

                label_losses: torch.Tensor = torch.Tensor(label_losses)

                loss: torch.Tensor = torch.dot(self._label_loss_weights, label_losses)

            with torch.no_grad():
                if self._use_cce_loss:
                    _probs = 1 - torch.sigmoid(out[:, 0, ...])
                    dice_sc = 1 - dice_round(_probs, 1 - msk_batch[:, 0, ...])
                else:
                    _probs = torch.sigmoid(out[:, 0, ...])
                    dice_sc = 1 - dice_round(_probs, msk_batch[:, 0, ...])

            losses_meter.update(loss.item(), img_batch.size(0))
            for j, loss_meter in enumerate(label_losses_meter):
                loss_meter.update(label_losses[j], img_batch.size(0))

            if self._use_cce_loss:
                ce_loss_meter.update(label_losses[5], img_batch.size(0))

            dices_meter.update(dice_sc, img_batch.size(0))

            iterator.set_description(
                f"epoch: {epoch};'"
                f" lr {self._lr_scheduler.get_last_lr()[-1]:.7f};"
                f" Total Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f});"
                f" Label Losses [{','.join(l.val() for l in label_losses_meter)}]"
                f" ([{','.join(l.avg() for l in label_losses_meter)}]);"
                f" Dice {dices_meter.val:.4f} ({dices_meter.avg:.4f});"
                + (f"CCE {ce_loss_meter.val:.4f} ({ce_loss_meter.avg:.4f});" if self._use_cce_loss else ""))

            self._update_weights(loss)

        self._lr_scheduler.step()

        log(f"epoch: {epoch};"
            f"lr {self._lr_scheduler.get_last_lr()[-1]:.7f};"
            f" Loss {losses_meter.avg:.4f};"
            f" Dice {dices_meter.avg:.4f};"
            + (f"CCE {ce_loss_meter.avg:.4f};" if self._use_cce_loss else ""))
