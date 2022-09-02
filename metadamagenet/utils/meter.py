class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self._val: float = 0
        self._avg: float = 0
        self._sum: float = 0
        self._count: float = 0

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, value: float, n: int = 1) -> None:
        self._val = value
        self._sum += value * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def val(self) -> float:
        return self._val

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> float:
        return self._count

    @property
    def avg(self) -> float:
        return self._avg
