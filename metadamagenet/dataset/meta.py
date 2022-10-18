import random
from dataclasses import dataclass
from typing import List, Iterable, Type, Tuple

from torch.utils.data import DataLoader

from .dataset import ImageData, Xview2Dataset


@dataclass
class Task:
    """
    support: contains support-k-shot samples
    query: contains query-k-shot samples
    """
    name: str
    support: DataLoader
    query: DataLoader


@dataclass
class TaskSet:
    tasks: List[Task]

    def __iter__(self):
        return self.tasks.__iter__()

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item: int) -> Task:
        return self.tasks[item]


class MetaDataLoader(Iterable):
    def __init__(self,
                 dataset_class: Type[Xview2Dataset],
                 tasks: List[Tuple[str, List[ImageData]]],
                 task_set_size: int,
                 support_shots: int,
                 query_shots: int,
                 batch_size: int = 1):
        self._dataset_class: Type[Xview2Dataset] = dataset_class
        self._tasks: List[Tuple[str, List[ImageData]]] = tasks
        self._task_set_size: int = task_set_size
        self._support_shots: int = support_shots
        self._query_shots: int = query_shots
        self._batch_size: int = batch_size
        self._i: int = 0

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self) -> int:
        return len(self._tasks) // self._task_set_size

    @property
    def total_tasks(self) -> int:
        return len(self) * self._task_set_size

    @property
    def task_set_size(self) -> int:
        return self._task_set_size

    def __next__(self) -> TaskSet:
        if self._i >= len(self):
            raise StopIteration
        self._i += 1

        tasks: List[Task] = []
        chosen_tasks: List[Tuple[str, List[ImageData]]] = random.sample(self._tasks, k=self._task_set_size)
        dataset: List[ImageData]
        for name, dataset in chosen_tasks:
            mini_dataset: List[ImageData] = random.sample(dataset, k=(self._support_shots + self._query_shots))
            support_set: List[ImageData] = mini_dataset[:self._support_shots]
            query_set: List[ImageData] = mini_dataset[self._support_shots:]
            tasks.append(Task(
                name=name,
                support=DataLoader(
                    dataset=self._dataset_class(support_set),
                    batch_size=self._batch_size,
                    num_workers=1,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=False
                ),
                query=DataLoader(
                    dataset=self._dataset_class(query_set),
                    batch_size=self._batch_size,
                    num_workers=1,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=False
                )
            ))
        return TaskSet(tasks)
