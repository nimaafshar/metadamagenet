import abc
from typing import Sequence, Dict, TypeVar, Generic, Tuple, Union, Any

import torch
from torch import nn

__all__ = ('ImageCollection', 'Random', 'Transform', 'CollectionTransform', 'OnlyOn', 'OneOf')


def _assert_prob(prob: float) -> float:
    """
    assert if some float is a probability
    """
    if not 0 <= prob <= 1:
        raise ValueError(f"{prob} is not a valid probability")
    return prob


ImageCollection = Dict[str,  # key is data type for example: 'img' or 'img_pre' or 'msk'
                       torch.FloatTensor]  # batch of images for that data type

StateType = TypeVar('StateType')


class Transform(nn.Module, abc.ABC, Generic[StateType]):
    """
    a transform which can ba applied to a batch of images.
    input values are expected to be in [0,1].
    """

    @abc.abstractmethod
    def generate_state(self, input_shape: torch.Size) -> StateType:
        """
        determine transformation state parameters.
        parameter types can be different based on augmentation
        :return: augmentation state parameters
        """
        pass

    @abc.abstractmethod
    def forward(self, images: torch.FloatTensor, state: StateType) -> torch.FloatTensor:
        """
        :param images: batch of image data with shape (B,C,H,W)
                       input values are expected to be in [0,1]
        :param state: needed parameters for transforms
        :return: augmented batch of images
        """
        pass


class CollectionTransform(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, img_group: ImageCollection) -> ImageCollection:
        pass


class OnlyOn(nn.Module):
    def __init__(self, transform: Transform, *keys: str):
        super().__init__()
        self.transform: Transform = transform
        self._keys: Sequence[str] = keys

    def keys(self) -> Sequence[str]:
        return self._keys

    def forward(self, img_group: ImageCollection, state: StateType, apply: torch.BoolTensor) -> ImageCollection:
        """
        :param img_group: image collection
        :param state: transform state
        :param apply: bool tensor of shape (B,) which indicates whether to apply transform on each image or not
        :return: transformed batch
        """
        keys = self._keys if len(self._keys) > 0 else img_group.keys()
        key: str
        for key in keys:
            inputs: torch.FloatTensor = img_group[key]
            img_group[key] = ~apply * inputs + apply * self.transform(inputs, state)
        return img_group


class Random(CollectionTransform):
    def __init__(self, transform: Union[Transform, OnlyOn], p: float):
        """
        :param transform: transform to apply
        :param p: probability of transform
        """
        super().__init__()
        self.transform: OnlyOn = transform if isinstance(transform, OnlyOn) else OnlyOn(transform)
        self._p: float = _assert_prob(p)

    def forward(self, img_group: ImageCollection) -> ImageCollection:
        try:
            input_shape: torch.Size = next(iter(img_group.values())).size()
        except StopIteration:
            raise ValueError(f"img_group should not be empty. keys are {img_group.keys()}")
        state = self.transform.transform.generate_state(input_shape)
        apply: torch.BoolTensor = torch.rand((input_shape[0],)) <= self._p
        return self.transform(img_group, state, apply)

    def probability(self) -> float:
        return self._p


class OneOf(CollectionTransform):
    def __init__(self, *transforms: Tuple[Union[CollectionTransform, OnlyOn], float]):
        super().__init__()
        transforms, probs = zip(*transforms)
        assert len(transforms) > 0, "no transforms found"
        if not (isinstance(transforms[0], CollectionTransform) or isinstance(transforms[0], OnlyOn)):
            raise ValueError(f"unexpected type {type(transforms[0])}")
        for t in transforms[1:]:
            if not isinstance(t, type(transforms[0])):
                raise ValueError("all transforms should be of the same type")
        self.transforms: nn.ModuleList = nn.ModuleList(transforms)
        self._probs: probs

    def forward(self, img_group: ImageCollection) -> ImageCollection:
        if isinstance(self.transforms[0], OnlyOn):
            input_shape: torch.Size = next(iter(img_group.values())).size()
            applied_to: torch.BoolTensor = torch.BoolTensor(torch.zeros(input_shape[0]))
            r: OnlyOn
            prob: float
            for r, prob in zip(self.transforms, self._probs):
                state = r.transform.generate_state(input_shape)
                randoms: torch.BoolTensor = torch.rand(input_shape[0]) <= prob
                img_group = r(img_group,
                              state,
                              torch.logical_and(torch.logical_not(applied_to), randoms))
                applied_to = torch.logical_or(applied_to, randoms)
                if torch.all(applied_to):
                    break
        elif isinstance(self.transforms[0], CollectionTransform):
            transform: CollectionTransform
            prob: float
            for transform, prob in zip(self.transforms, self._probs):
                if torch.rand() <= prob:
                    img_group = transform(img_group)
                    break
        return img_group
