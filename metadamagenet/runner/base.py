from typing import Any
import abc


class Runner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self) -> Any:
        pass
