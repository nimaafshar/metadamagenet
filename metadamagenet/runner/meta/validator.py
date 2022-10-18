import logging
from typing import Optional, Dict, Callable

import emoji
import higher
import torch
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Metric, MeanMetric
from tqdm.autonotebook import tqdm

from ...logging import EmojiAdapter
from ...models import BaseModel
from ...dataset import MetaDataLoader, TaskSet, Task
from ..base import Runner

logger = EmojiAdapter(logging.getLogger())


class MetaValidator(Runner):
    def __init__(self,
                 model: BaseModel,
                 meta_dataloader: MetaDataLoader,
                 score: Metric,
                 inner_opt: Callable[[nn.Module, ], torch.optim.Optimizer],
                 n_inner_iter: int,
                 transform: Optional[nn.Module] = None,
                 loss: nn.Module = None,
                 device: Optional[torch.device] = None):
        self._device: torch.device
        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._model: BaseModel = model.to(self._device)
        self._transform: nn.Module = transform
        if self._transform is not None:
            self._transform = self._transform.to(self._device)
        self._meta_dataloader: MetaDataLoader = meta_dataloader
        self._loss: nn.Module = loss.to(self._device)

        self._score: Metric = score.to(self._device)
        self._n_inner_iter: int = n_inner_iter
        self._inner_optim: Callable[[nn.Module, ], torch.optim.Optimizer] = inner_opt

    def _prepare_batch(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._device is not None:
            data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in
                          data_batch.items()}
        with torch.no_grad():
            if self._transform is not None:
                data_batch = self._transform(data_batch)
            return self._model.preprocess(data_batch)

    def run(self) -> float:
        # Crucially in our testing procedure here, we do *not* fine-tune
        # the model during testing for simplicity.
        # Most research papers using MAML for this task do an extra
        # stage of fine-tuning here that should be added if you are
        # adapting this code for research.
        logger.info(f':arrow_forward: starting to validate model {self._model.name()} in meta-learning setup')
        logger.info(f'meta-steps: {len(self._meta_dataloader)}')

        self._model.train()
        self._score.reset()
        total_support_loss: MeanMetric = MeanMetric().to(device=self._device)
        total_support_score: MeanMetric = MeanMetric().to(device=self._device)
        total_query_loss: MeanMetric = MeanMetric().to(device=self._device)
        total_query_score: MeanMetric = MeanMetric().to(device=self._device)

        with tqdm(total=self._meta_dataloader.total_tasks,
                  leave=False,
                  desc=emoji.emojize(":fearful: Task Set Meta-Validation", language='alias')) as progress_bar:
            task_set: TaskSet
            for task_set in self._meta_dataloader:
                inner_optim: Optimizer = self._inner_optim(self._model)
                task: Task
                for task in task_set:
                    with higher.innerloop_ctx(self._model,
                                              inner_optim,
                                              track_higher_grads=False) as (f_model, diff_optim):
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        inputs: torch.Tensor
                        targets: torch.Tensor
                        data_batch: Dict[str, torch.Tensor]

                        self._score.reset()
                        for k in range(1, self._n_inner_iter + 1):
                            support_loss_sum: torch.Tensor = 0
                            for data_batch in task.support:
                                inputs, targets = self._prepare_batch(data_batch)
                                outputs: torch.Tensor = f_model(inputs)
                                support_loss_sum += self._loss(outputs, targets)
                                with torch.no_grad():
                                    activated_outputs: torch.Tensor = self._model.activate(outputs)
                                    self._score.update(activated_outputs, targets)
                                del inputs, targets, outputs, activated_outputs
                            support_loss = support_loss_sum / len(task.support)
                            diff_optim.step(support_loss)

                            logger.info("%s", {
                                "task": task.name,
                                "mode": "adapt",
                                "k": f"{k}/{self._n_inner_iter}",
                                "loss": support_loss.item(),
                                "score": self._score.compute().item()
                            })
                            total_support_loss.update(support_loss.item())
                            total_support_score.update(self._score.compute().item())
                            progress_bar.set_postfix({
                                "sup_loss": total_support_loss.compute().item(),
                                "sup_sc": total_support_score.compute().item()
                            })

                        # The query loss and score induced by these parameters.
                        self._score.reset()
                        query_loss_sum: torch.Tensor = 0
                        for data_batch in task.query:
                            inputs, targets = self._prepare_batch(data_batch)
                            outputs = f_model(inputs).detach()
                            query_loss_sum += self._loss(outputs, targets).detach()
                            with torch.no_grad():
                                activated_outputs: torch.Tensor = self._model.activate(outputs)
                                self._score.update(activated_outputs, targets).detach()
                            del inputs, targets, outputs, activated_outputs
                        query_loss: torch.Tensor = (query_loss_sum / len(task.query))
                        logger.info("%s", {
                            "task": task.name,
                            "mode": "check",
                            "loss": query_loss.item(),
                            "score": self._score.compute().item()
                        })
                        total_query_loss.update(query_loss.item())
                        total_query_score.update(self._score.compute().item())
                        progress_bar.set_postfix({
                            "que_loss": total_query_loss.compute().item(),
                            "que_sc": total_query_score.compute().item()
                        })
                    progress_bar.update(1)
        logger.info("%s", {
            "support_loss": total_support_loss.compute().item(),
            "support_score": total_support_score.compute().item(),
            "query_loss": total_query_loss.compute().item(),
            "query_score": total_query_score.compute().item()
        })
        return total_query_score.compute().item()
