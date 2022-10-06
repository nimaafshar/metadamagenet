from typing import Optional, Union

from tqdm.autonotebook import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from .base import Runner
from ..augment import TestTimeAugmentor
from torchmetrics import Metric
from ..logging import log
from ..models import BaseModel, ModelAggregator


class Validator(Runner):
    def __init__(self,
                 model: Union[BaseModel, ModelAggregator],
                 dataloader: DataLoader,
                 score: Metric,
                 transform: Optional[nn.Module] = None,
                 loss: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 test_time_augmentor: Optional[TestTimeAugmentor] = None
                 ):

        self._device: torch.device
        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._model: Union[BaseModel, ModelAggregator] = model.to(self._device)
        self._transform: nn.Module = transform
        if self._transform is not None:
            self._transform = self._transform.to(self._device)
        self._dataloader: DataLoader = dataloader
        self._dataloader.pin_memory = True
        self._loss: Optional[nn.Module] = loss
        if self._loss is not None:
            self._loss = self._loss.to(self._device)

        self._score: Metric = score.to(self._device)

        self._test_time_augmentor: Optional[TestTimeAugmentor] = test_time_augmentor
        if self._test_time_augmentor is not None and isinstance(self._model, ModelAggregator):
            raise ValueError("test time augment cannot be used with model aggregators")

        if self._loss is not None and isinstance(self._model, ModelAggregator):
            raise ValueError("loss calculation cannot be used with model aggregators")

    def run(self) -> float:
        """
        validate model with given data
        :return: score
        """
        log(f':arrow_forward: starting to validate model {self._model.name()} ...')
        log(f'steps: {len(self._dataloader)}')
        self._model.eval()
        loss_mean: MeanMetric = MeanMetric().to(self._device)
        self._score.reset()
        iterator = tqdm(self._dataloader, leave=False)

        with torch.no_grad():
            i: int
            for i, data_batch in enumerate(iterator):
                if self._device is not None:
                    data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in data_batch.items()}

                if self._transform is not None:
                    data_batch = self._transform(data_batch)

                inputs: torch.Tensor
                targets: torch.Tensor
                inputs, targets = self._model.preprocess(data_batch)
                activated_outputs: torch.Tensor
                if isinstance(self._model, BaseModel):
                    outputs: torch.Tensor
                    if self._test_time_augmentor is not None:
                        augmented_inputs: torch.Tensor = self._test_time_augmentor.augment(inputs)
                        augmented_outputs: torch.Tensor = self._model(augmented_inputs)
                        outputs: torch.Tensor = self._test_time_augmentor.aggregate(augmented_outputs)
                    else:
                        outputs: torch.Tensor = self._model(inputs)

                    if self._loss is not None:
                        loss = self._loss(outputs, targets)
                        loss_mean.update(loss)

                    activated_outputs = self._model.activate(outputs)
                else:
                    assert isinstance(self._model, ModelAggregator)
                    activated_outputs = self._model(inputs)

                current_score: torch.Tensor = self._score(activated_outputs, targets)
                iterator.set_postfix({
                    "loss": loss.item() if self._loss is not None else "--",
                    "score": current_score.item()
                })

            log(f"Validation Results: loss:{loss_mean.compute().item() if self._loss is not None else '--'} "
                f"score:{self._score.compute().item()}")
            return self._score.compute().item()
