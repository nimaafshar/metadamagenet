from typing import Optional

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..augment import TestTimeAugmentor
from ..metrics import ImageMetric
from ..wrapper import ModelWrapper
from ..logging import log
from ..losses import MonitoredImageLoss, Monitored


class Validator:
    def __init__(self,
                 model: nn.Module,
                 model_wrapper: ModelWrapper,
                 dataloader: DataLoader,
                 loss: Optional[nn.Module],
                 score: ImageMetric,
                 device: Optional[torch.device] = None,
                 test_time_augmentor: Optional[TestTimeAugmentor] = None
                 ):
        self._model: nn.Module = model
        self._wrapper: ModelWrapper = model_wrapper
        self._dataloader: DataLoader = dataloader
        self._loss: Optional[MonitoredImageLoss] = loss if loss is None or isinstance(loss, MonitoredImageLoss) \
            else Monitored(loss)
        self._score: ImageMetric = score
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
        if self._loss is not None:
            self._loss.reset()
        self._score.reset()
        iterator = tqdm(self._dataloader)

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

                with torch.no_grad():
                    if self._loss is not None:
                        self._loss(outputs, targets)

                    activated_outputs: torch.Tensor = self._wrapper.apply_activation(outputs)
                    self._score.update_batch(activated_outputs, targets)

                iterator.set_postfix({
                    "loss": self._loss.status_till_here if self._loss is not None else "--",
                    "score": self._score.status_till_here
                })

            log(f"Validation Results: loss:{self._loss.status_till_here if self._loss is not None else '--'} "
                f"score:{self._score.status_till_here}")
            return self._score.till_here
