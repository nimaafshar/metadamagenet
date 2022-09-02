import abc
import numpy as np
import numpy.typing as npt
import torch


class TestTimeAugmentor(abc.ABC):
    def augment(self, img_batch: torch.FloatTensor) -> torch.FloatTensor:
        """
        augment a batch of images. this may change batch size
        :param img_batch: torch.FloatTensor of shape (original batch size,channels,height,width)
        :return: torch.FloatTensor of shape (augmented batch size,channels,height,width)
        """
        augmented: npt.NDArray[np.float32] = self._augment(
            img_batch
            .numpy()
            .transpose(0, 2, 3, 1)
        ).transpose(0, 3, 1, 2)
        return torch.from_numpy(augmented)

    @abc.abstractmethod
    def _augment(self, img_batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        implementation of the augment operation
        :param img_batch: np.ndarray[np.float32] of shape (original batch size,height,width,channels)
        :return: np.ndarray[np.float32] of shape (augmented batch size,height,width,channels)
        """
        pass

    def aggregate(self, augmented_img_batch: torch.FloatTensor) -> torch.FloatTensor:
        """
        aggregate augmentations in a batch of images. this method reverts batch size to its original value
        :param augmented_img_batch: torch.FloatTensor of shape (augmented batch size,channels,height,width)
        :return: :return: torch.FloatTensor of shape (original batch size,channels,height,width)
        """
        aggregated: npt.NDArray[np.float32] = self._aggregate(
            augmented_img_batch
            .numpy()
            .transpose(0, 2, 3, 1)
        ).transpose(0, 3, 1, 2)
        return torch.from_numpy(aggregated)

    @abc.abstractmethod
    def _aggregate(self, augmented_img_batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        implementation of the aggregate operation
        :param augmented_img_batch: np.ndarray[np.float32] of shape (augmented batch size,height,width,channels)
        :return: np.ndarray[np.float32] of shape (original batch size,height,width,channels)
        """
        pass


class FourFlips(TestTimeAugmentor):
    def _augment(self, img_batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        augment using combinations of top-down and left-right flips
        """
        return np.vstack(
            (img_batch,  # original
             img_batch[:, ::-1, ...],  # top-down flip
             img_batch[:, :, ::-1, ...],  # left-right flip
             img_batch[:, ::-1, ::-1, ...]  # flip along both x and y-axis (180 rotation)
             )
        )

    def _aggregate(self, augmented_img_batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        revere flips and aggregate augmentations of every image in the batch by taking their mean
        """
        reshaped = augmented_img_batch.reshape((4, augmented_img_batch.shape[0] // 4) + augmented_img_batch.shape[1:])
        return np.stack(
            (reshaped[0, ...],  # original
             reshaped[1, :, ::-1, ...],  # top-down flip
             reshaped[2, :, :, ::-1, ...],  # left-right flip
             reshaped[3, :, ::-1, ::-1, ...]  # flip along both x and y-axis (180 rotation)
             )
        ).mean(axis=0)
