import abc

import torch


class Metric:
    """
    Interface of a metric
    """

    @abc.abstractmethod
    def till_here(self) -> float:
        """
        metric value till here
        """
        pass

    @abc.abstractmethod
    def status_till_here(self) -> str:
        """
        metric value till here formatted as string
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        reset recorded state
        """
        pass


class ImageMetric(Metric, torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        calculate and update metric based on a batch of outputs and a batch of targets
        :param outputs: tensor of shape (B,C,H,W)
        :param targets: tensor of shape (B,C,H,W)
        :return: metric value for this batch
        """
