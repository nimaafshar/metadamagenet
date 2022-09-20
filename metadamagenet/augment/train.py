import dataclasses
import random
from typing import Tuple, Type, Any, Optional, Sequence, Dict

import numpy.typing as npt


class RandomCrop:
    """
    crop image with a random square.
    does multiple tries, selects the one who is covering more of the white pixels in the mask
    """

    @dataclasses.dataclass
    class RandomCropParams:
        crop_size: int
        try_count: int
        x0: Optional[int] = None
        y0: Optional[int] = None
        batch: bool = False

    ParamType = RandomCropParams

    def __init__(self, default_crop_size: int, size_change_probability: float, crop_size_range: Tuple[int, int],
                 try_range: Tuple[int, int], apply_to: Optional[Sequence[str]] = None,
                 scoring_weights: Optional[Dict[str, float]] = None):
        """
        :param default_crop_size: default crop size
        :param size_change_probability: 1 - probability of crop size being changed randomly
        :param crop_size_range: range of crop size
        :param try_range: range for the number of tries to select the best crop randomly
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(apply_to)
        self._default_crop_size: int = default_crop_size
        if not 0 <= size_change_probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")
        self._size_change_probability: float = size_change_probability
        self._crop_size_range: Tuple[int, int] = crop_size_range
        self._try_range: Tuple[int, int] = try_range
        self._scoring_weights: Optional[Dict[str, float]] = scoring_weights

    @property
    def _crop_size(self) -> int:
        if random.random() > self._size_change_probability:
            return random.randint(*self._crop_size_range)
        else:
            return self._default_crop_size

    @staticmethod
    def score(msk: npt.NDArray, crop_size: int, x0: int, y0: int) -> int:
        return msk[y0:y0 + crop_size, x0:x0 + crop_size].sum()

    @staticmethod
    def _score_on_batch(msk_batch: Dict[str, npt.NDArray], crop_size: int, x0: int, y0: int,
                        weights: Dict[str, float]) -> int:
        score: int = 0
        for key, weight in weights.items():
            msk = msk_batch.get(key)
            if msk is not None:
                score += RandomCrop.score(msk, crop_size, x0, y0) * weight
        return score

    def _determine_params(self) -> ParamType:
        return RandomCrop.RandomCropParams(
            crop_size=self._crop_size,
            try_count=random.randint(*self._try_range)
        )

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return img[params.y0: params.y0 + params.crop_size, params.x0: params.x0 + params.crop_size, ...]

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        params.batch = False

        params = self._best_crop(msk, params)

        img = self._transform(img, params)
        msk = self._transform(msk, params)
        return img, msk

    def _best_crop_batch(self, img_batch: Dict[str, npt.NDArray], params: ParamType) -> ParamType:
        img = next(iter(img_batch.values()))
        best_x0 = random.randint(0, img.shape[1] - params.crop_size)
        best_y0 = random.randint(0, img.shape[0] - params.crop_size)
        best_score = -1

        weights = self._scoring_weights if self._scoring_weights is not None else {key: 1 for key in img_batch}

        for _ in range(params.try_count):
            x0 = random.randint(0, img.shape[1] - params.crop_size)
            y0 = random.randint(0, img.shape[0] - params.crop_size)
            score = RandomCrop._score_on_batch(img_batch, params.crop_size, x0, y0, weights)
            if score > best_score:
                best_score = score
                best_x0 = x0
                best_y0 = y0
        params.x0 = best_x0
        params.y0 = best_y0
        return params

    @staticmethod
    def _best_crop(msk: npt.NDArray, params: ParamType) -> ParamType:
        best_x0 = random.randint(0, msk.shape[1] - params.crop_size)
        best_y0 = random.randint(0, msk.shape[0] - params.crop_size)
        best_score = -1

        for _ in range(params.try_count):
            x0 = random.randint(0, msk.shape[1] - params.crop_size)
            y0 = random.randint(0, msk.shape[0] - params.crop_size)
            score = RandomCrop.score(msk, params.crop_size, x0, y0)
            if score > best_score:
                best_score = score
                best_x0 = x0
                best_y0 = y0
        params.x0 = best_x0
        params.y0 = best_y0
        return params

    def apply_batch(self, img_batch: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray], bool]:
        params = self._determine_params()
        params.batch = True

        params = self._best_crop_batch(img_batch, params)

        keys = self._apply_to if self._apply_to is not None else img_batch.keys()

        for key in keys:
            img_batch[key] = self._transform(img_batch[key], params)

        return img_batch, True
