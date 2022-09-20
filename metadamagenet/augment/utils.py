from typing import Tuple

import torch


def random_float_tensor(size: torch.Size, range_: Tuple[float, float], device: torch.device) -> torch.FloatTensor:
    return torch.rand(size, device=device, requires_grad=False) * (range_[1] - range_[0]) + range_[0]
