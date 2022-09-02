import abc

import torch


class TestTimeAugmentor(abc.ABC):

    @abc.abstractmethod
    def augment(self, img_batch: torch.FloatTensor) -> torch.FloatTensor:
        """
        augment a batch of images. this may change batch size
        :param img_batch: torch.FloatTensor of shape (original batch size,channels,height,width)
        :return: torch.FloatTensor of shape (augmented batch size,channels,height,width)
        """
        pass

    @abc.abstractmethod
    def aggregate(self, augmented_img_batch: torch.FloatTensor) -> torch.FloatTensor:
        """
        aggregate augmentations in a batch of images. this method reverts batch size to its original value
        :param augmented_img_batch: torch.FloatTensor of shape (augmented batch size,channels,height,width)
        :return: :return: torch.FloatTensor of shape (original batch size,channels,height,width)
        """
        pass


class FourFlips(TestTimeAugmentor):
    def augment(self, img_batch: torch.FloatTensor) -> torch.FloatTensor:
        """
        augment using combinations of top-down and left-right flips
        """
        transposed: torch.FloatTensor = img_batch.permute(0, 2, 3, 1)
        return torch.vstack(
            (transposed,  # original
             transposed.flip((1,)),  # top-down flip
             transposed.flip((2,)),  # left-right flip
             transposed.flip((1, 2))  # flip along both x and y-axis (180 rotation)
             )
        ).permute(0, 3, 1, 2)

    def aggregate(self, augmented_img_batch: torch.FloatTensor) -> torch.FloatTensor:
        """
        revere flips and aggregate augmentations of every image in the batch by taking their mean
        """
        transposed = augmented_img_batch.permute(0, 2, 3, 1)
        reshaped = transposed.reshape((4, transposed.size(0) // 4) + transposed.size()[1:])
        return torch.stack(
            (reshaped[0],  # original
             reshaped[1].flip((1,)),  # top-down flip
             reshaped[2].flip((2,)),  # left-right flip
             reshaped[3].flip((1, 2)),  # flip along both x and y-axis (180 rotation)
             )
        ).mean(axis=0).permute(0, 3, 1, 2)
