from typing import Optional

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from metadamagenet.augment import TestTimeAugmentor
from metadamagenet.metrics import Score, AverageMeter
from metadamagenet.wrapper import ModelWrapper
from metadamagenet.logging import log
from metadamagenet.losses import MonitoredLoss


class Validator:
    def __init__(self,
                 model: nn.Module,
                 model_wrapper: ModelWrapper,
                 dataloader: DataLoader,
                 loss: Optional[MonitoredLoss],
                 score: Score,
                 device: Optional[torch.device] = None,
                 test_time_augmentor: Optional[TestTimeAugmentor] = None
                 ):
        self._model: nn.Module = model
        self._wrapper: ModelWrapper = model_wrapper
        self._dataloader: DataLoader = dataloader
        self._loss: Optional[MonitoredLoss] = loss
        self._score: Score = Score
        self._device: Optional[torch.device] = device
        self._test_time_augmentor: Optional[TestTimeAugmentor] = test_time_augmentor

    def validate(self) -> float:
        """
        validate model with given data
        :return: score
        """
        log(f':arrow_forward: starting to validate model {self._wrapper.model_name} ...')
        log(f'steps: {len(self._dataloader)}')
        self._model.eval()
        iterator = tqdm(self._dataloader)

        loss_meter: AverageMeter = AverageMeter()

        with torch.no_grad():
            i: int
            inputs: torch.Tensor
            targets: torch.Tensor
            for i, (inputs, targets) in enumerate(iterator):
                if self._device is not None:
                    inputs = inputs.to(device=self._device, non_blocking=True)
                    targets = targets.to(device=self._device, non_blocking=True)
                outputs: torch.Tensor
                if self._test_time_augmentor is not None:
                    augmented_inputs: torch.Tensor = self._test_time_augmentor.augment(inputs)
                    augmented_outputs: torch.Tensor = self._model(augmented_inputs)
                    outputs: torch.Tensor = self._test_time_augmentor.aggregate(augmented_outputs)
                else:
                    outputs: torch.Tensor = self._model(inputs)

                loss_status: str = "--"
                if self._loss is not None:
                    loss_value: torch.Tensor = self._loss(outputs, targets)
                    if self._loss.monitored:
                        loss_status = self._loss.last_values()
                    else:
                        loss_meter.update(loss_value.item(), inputs.size(0))
                        loss_status = loss_meter.status

                activated_outputs: torch.Tensor = self._wrapper.apply_activation(outputs)
                self._score.update(activated_outputs, targets)
                score_status: str = self._score.status()
                iterator.set_postfix({"loss": loss_status, "score": score_status})

            loss_status: str = "--"
            if self._loss is not None:
                if self._loss.monitored:
                    loss_status = self._loss.aggregate()
                else:
                    loss_status = loss_meter.avg_status

            score_status: str = self._score.avg_status()
            log(f"Validation Results: loss:{loss_status} score:{score_status}")
            return self._score.avg
