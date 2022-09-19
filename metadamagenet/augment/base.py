import abc
from typing import Sequence, Dict, Optional, TypeVar, Generic, List

import torch
from torch import nn


def assert_prob(prob: float) -> float:
    """
    assert if some float is a probability
    """
    if not 0 <= prob < 1:
        raise ValueError(f"{prob} is not a valid probability")
    return prob


ImageCollection = Dict[str,  # key is data type for example: 'img' or 'img_pre' or 'msk'
                       torch.Tensor]  # batch of images for that data type

StateType = TypeVar('StateType')


class Transform(nn.Module, abc.ABC, Generic[StateType]):
    """a transform which can ba applied to a dict of str to batch of images
        input values are expected to be in [0,1]
    """

    def __init__(self, apply_to: Optional[Sequence[str]] = None):
        """
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all keys.
        """
        super().__init__()
        self._apply_to: Optional[Sequence[str]] = apply_to

    @abc.abstractmethod
    def transform(self, images: torch.Tensor, state: StateType) -> torch.Tensor:
        """
        :param images: batch of image data with shape (B,C,H,W)
                       input values are expected to be in [0,1]
        :param state: needed parameters for transforms
        :return: augmented batch of images
        """
        pass

    def forward(self, img_group: ImageCollection) -> ImageCollection:
        keys = self._apply_to if self._apply_to is not None else img_group.keys()
        for key in keys:
            inputs: torch.Tensor = img_group[key]
            assert len(inputs.size()) == 4, "inputs dimension is not 4"
            outputs: torch.Tensor = self.transform(inputs, None)
            assert outputs.size() == inputs.size(), "outputs dimension is not the same as inputs"
            img_group[key] = outputs
        return img_group


class RandomTransform(Transform, abc.ABC):
    def __init__(self, apply_to: Optional[Sequence[str]] = None, p: float = 0):
        super().__init__(apply_to)
        self._p: float = assert_prob(p)

    @abc.abstractmethod
    def generate_random_state(self, input_shape: torch.Size) -> StateType:
        """
        determine augmentation state parameters.
        parameter types can be different based on augmentation
        :return: augmentation state parameters
        """
        pass

    def forward_to(self,
                   img_group: ImageCollection,
                   state: StateType,
                   do_transform: torch.BoolTensor) -> ImageCollection:
        keys = self._apply_to if self._apply_to is not None else img_group.keys()
        for key in keys:
            inputs: torch.Tensor = img_group[key]
            assert len(inputs.size()) == 4, "inputs dimension is not 4"
            outputs: torch.Tensor = self.transform(inputs, state)
            assert outputs.size() == inputs.size(), "outputs dimension is not the same as inputs"
            img_group[key] = torch.logical_not(do_transform) * inputs + do_transform * outputs

    def forward(self, img_group: ImageCollection) -> ImageCollection:
        input_shape: torch.Size = next(iter(img_group.values())).size()
        state: StateType = self.generate_random_state(input_shape)
        do_transform: torch.BoolTensor = torch.rand_like(input_shape[0]) > self._p
        return self.forward_to(img_group, state, do_transform)

    def prob(self) -> float:
        return self._p


class OneOf(Transform):
    def __init__(self, transforms: List[RandomTransform], p: float = 0):
        super().__init__(apply_to=None)
        self.transforms: nn.ModuleList = nn.ModuleList(transforms)

    def transform(self, images: torch.Tensor, state: StateType) -> torch.Tensor:
        raise NotImplemented

    def forward(self, img_group: ImageCollection) -> ImageCollection:
        if not torch.rand() > self._p:
            return img_group

        input_shape: torch.Size = next(iter(img_group.values())).size()
        applied_to: torch.BoolTensor = torch.BoolTensor(torch.zeros(input_shape[0]))

        t: RandomTransform
        for t in self.transforms:
            state = t.generate_random_state(input_shape[0])
            randoms: torch.BoolTensor = torch.rand(input_shape[0]) > t.prob()
            img_group = t.forward_to(img_group,
                                     state,
                                     torch.logical_and(torch.logical_not(applied_to), randoms))
            applied_to = torch.logical_or(applied_to, randoms)
            if torch.all(applied_to):
                break
        return img_group
