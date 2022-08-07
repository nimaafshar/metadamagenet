import random
from typing import Union

import numpy as np
from torch import nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.backends import cudnn
from tqdm import tqdm

from .trainer import Trainer, TrainingConfig
from src.optim import AdamW
from src.util.utils import AverageMeter, dice
from src.losses import dice_round, ComboLoss
from src.logs import log


class LocalizationTrainer(Trainer):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._model: nn.Module = self._get_model()
        self._optimizer: torch.optim.Optimizer = AdamW(self._model.parameters(),
                                                       lr=0.00015,
                                                       weight_decay=1e-6)
        self._lr_scheduler: torch.optim.lr_scheduler.MultiStepLR = MultiStepLR(self._optimizer,
                                                                               milestones=[5, 11, 17, 25, 33, 47, 50,
                                                                                           60, 70, 90, 110, 130, 150,
                                                                                           170, 180, 190],
                                                                               gamma=0.5)
        self._model: nn.Module = nn.DataParallel(self._model).cuda()
        self._seg_loss: ComboLoss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()
        self._evaluation_dice_thr: float = 0.5

    def _setup(self):
        super(LocalizationTrainer, self)._setup()
        # vis_dev = sys.argv[2]

        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

        cudnn.benchmark = True
        np.random.seed(self._config.model_config.seed + 545)
        random.seed(self._config.model_config.seed + 454)

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
        losses: AverageMeter = AverageMeter()
        dices: AverageMeter = AverageMeter()

        self._model.train()

        iterator = tqdm(self._train_data_loader)

        for i, (img_batch, msk_batch) in enumerate(iterator):
            img_batch = img_batch.cuda(non_blocking=True)
            msk_batch = msk_batch.cuda(non_blocking=True)

            out = self._model(img_batch)

            loss = self._seg_loss(out, msk_batch)

            with torch.no_grad():
                _probs = torch.sigmoid(out[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, msk_batch[:, 0, ...])

            losses.update(loss.item(), img_batch.size(0))

            dices.update(dice_sc, img_batch.size(0))

            # TODO: test get_lr() method
            iterator.set_description(
                f"epoch: {epoch};'"
                f" lr {self._lr_scheduler.get_lr()[-1]:.7f};"
                f" Loss {losses.val:.4f} ({loss.avg:.4f});"
                f" Dice {dices.val:.4f} ({dices.avg:.4f})")

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
            self._optimizer.step()

        self._lr_scheduler.step(epoch)

        log(f"epoch: {epoch}; lr {self._lr_scheduler.get_lr()[-1]:.7f}; Loss {losses.avg:.4f}; Dice {dices.avg:.4f}")
