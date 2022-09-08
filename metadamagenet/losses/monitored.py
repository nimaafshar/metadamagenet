import abc

from torch import nn


class MonitoredLoss(nn.Module, abc.ABC):
    """
    monitored values should be updated in every call of forward() method
    """

    @property
    @abc.abstractmethod
    def monitored(self) -> bool:
        """
        :return: is this instance doing monitoring
        """
        pass

    @abc.abstractmethod
    def last_values(self) -> str:
        """
        :return: last monitored values formatted in string
        """
        pass

    @abc.abstractmethod
    def aggregate(self) -> str:
        """
        aggregate results, format them as string and reset attributes
        :return:aggregated results as string
        """
        pass
