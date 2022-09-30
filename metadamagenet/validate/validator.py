from typing import Optional

from tqdm.autonotebook import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..augment import TestTimeAugmentor
from torchmetrics import Metric
from ..wrapper import ModelWrapper
from ..logging import log
from ..losses import MonitoredImageLoss, Monitored
from ..dataset import ImagePreprocessor


class Validator:
    def __init__(self,
                 model: nn.Module,
                 model_wrapper: ModelWrapper,
                 dataloader: DataLoader,
                 preprocessor: ImagePreprocessor,
                 loss: Optional[nn.Module],
                 score: Metric,
                 device: Optional[torch.device] = None,
                 test_time_augmentor: Optional[TestTimeAugmentor] = None
                 ):
        self._device: Optional[torch.device] = device

        self._model: nn.Module = model
        if self._device is not None:
            self._model = self._model.to(device)

        self._wrapper: ModelWrapper = model_wrapper
        self._dataloader: DataLoader = dataloader

        self._preprocessor: ImagePreprocessor = preprocessor
        if self._device is not None:
            self._preprocessor = self._preprocessor.to(self._device)

        self._loss: Optional[MonitoredImageLoss] = loss if loss is None or isinstance(loss, MonitoredImageLoss) \
            else Monitored(loss)
        if self._loss is not None and self._device is not None:
            self._loss = self._loss.to(self._device)

        self._score: Metric = score
        if self._device is not None:
            self._score = self._score.to(self._device)

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
        iterator = tqdm(self._dataloader, leave=False)

        with torch.no_grad():
            i: int
            for i, data_batch in enumerate(iterator):
                if self._device is not None:
                    data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in data_batch.items()}

                inputs: torch.Tensor
                targets: torch.Tensor
                inputs, targets = self._preprocessor(data_batch)

                outputs: torch.Tensor
                if self._test_time_augmentor is not None:
                    augmented_inputs: torch.Tensor = self._test_time_augmentor.augment(inputs)
                    augmented_outputs: torch.Tensor = self._model(augmented_inputs)
                    outputs: torch.Tensor = self._test_time_augmentor.aggregate(augmented_outputs)
                else:
                    outputs: torch.Tensor = self._model(inputs)

                if self._loss is not None:
                    self._loss(outputs, targets)

                activated_outputs: torch.Tensor = self._wrapper.apply_activation(outputs)
                current_score: torch.Tensor = self._score(activated_outputs, targets)

                iterator.set_postfix({
                    "loss": self._loss.status_till_here() if self._loss is not None else "--",
                    "score": current_score.item()
                })

            log(f"Validation Results: loss:{self._loss.status_till_here() if self._loss is not None else '--'} "
                f"score:{self._score.compute().item()}")
            return self._score.compute().item()
