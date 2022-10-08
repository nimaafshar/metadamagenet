from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torch import nn

from metadamagenet.utils import set_random_seeds
from metadamagenet.dataset import LocalizationDataset, ClassificationDataset
from metadamagenet.models import Localizer, Classifier
from metadamagenet.models.unet import SeResnext50Unet
from metadamagenet.losses import WeightedSum, BinaryFocalLoss2d, BinaryDiceLossWithLogits, DiceLoss, FocalLoss2d, \
    SegmentationCCE
from metadamagenet.metrics import xview2
from metadamagenet.runner import Trainer, ValidationInTrainingParams
from metadamagenet.augment import Random, VFlip, Rotate90, Shift, RotateAndScale, BestCrop, OneOf, RGBShift, HSVShift, \
    Clahe, GaussianNoise, Blur, Saturation, Brightness, Contrast, ElasticTransform, Dilation

train_dir = Path('/datasets/xview2/train')
test_dir = Path('/datasets/xview2/test')


class SeResnext50Localizer(Localizer[SeResnext50Unet]):
    pass


class SeResnext50Classifier(Classifier[SeResnext50Unet]):
    pass


def train_localizer(seed: int):
    set_random_seeds(seed + 123)
    transform = nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.1),
        Random(RotateAndScale(), p=0.1),
        BestCrop(samples=5, dsize=(512, 512), size_range=(0.45, 0.55)),
        Random(RGBShift().only_on('img'), p=0.01),
        Random(HSVShift().only_on('img'), p=0.01),
        OneOf(
            (OneOf(
                (Clahe().only_on('img'), 0.01),
                (GaussianNoise().only_on('img'), 0.01),
                (Blur().only_on('img'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img'), 0.01),
                (Brightness().only_on('img'), 0.01),
                (Contrast().only_on('img'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img'), p=0.001)
    )
    model = SeResnext50Localizer(SeResnext50Unet(pretrained_backbone=True))
    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180,
                                           190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='0',
        seed=0,
        dataloader=DataLoader(
            dataset=LocalizationDataset(train_dir),
            batch_size=15,
            shuffle=True,
            drop_last=True,
            num_workers=5,
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (BinaryDiceLossWithLogits(), 1.0),
            (BinaryFocalLoss2d(), 10.0)
        ),
        score=xview2.localization_score,
        epochs=150,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=1.1,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                shuffle=True,
                drop_last=False,
                num_workers=5,
            ),
            transform=None,
            interval=2,
        )
    ).run()


def tune_localizer(seed: int):
    set_random_seeds(432 + seed)
    transform = nn.Sequential(
        Random(VFlip(), p=0.45),
        Random(Rotate90(), p=0.9),
        Random(Shift(), p=.05),
        Random(RotateAndScale(), p=0.05),
        BestCrop(samples=5, dsize=(512, 512), size_range=(0.45, 0.55)),
        OneOf(
            (RGBShift().only_on('img'), 0.01),
            (HSVShift().only_on('img'), 0.01)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img'), 0.01),
                (GaussianNoise().only_on('img'), 0.01),
                (Blur().only_on('img'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img'), 0.01),
                (Brightness().only_on('img'), 0.01),
                (Contrast().only_on('img'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img'), p=0.001)
    )

    model = SeResnext50Localizer.from_checkpoint(version='0', seed=0)
    optimizer = AdamW(model.parameters(), lr=0.00004, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                           90, 110, 130, 150, 170, 180, 190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='0',
        seed=0,
        dataloader=DataLoader(
            dataset=LocalizationDataset(train_dir),
            batch_size=15,
            num_workers=6,
            shuffle=True,
            drop_last=True,
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (BinaryDiceLossWithLogits(), 1.0),
            (BinaryFocalLoss2d(), 10.0)
        ),
        score=xview2.localization_score,
        epochs=12,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=1.1,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False,
            ),
            transform=None,
            interval=1,
        )
    )


def train_classifier(seed: int):
    set_random_seeds(1234 + seed)

    transform = nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.05),
        Random(Shift(), p=.2),
        Random(RotateAndScale(), p=0.8),
        BestCrop(samples=10, dsize=(512, 512), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.04),
            (RGBShift().only_on('img_post'), 0.04),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.04),
            (HSVShift().only_on('img_post'), 0.04),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.04),
                (GaussianNoise().only_on('img_pre'), 0.04),
                (Blur().only_on('img_pre'), 0.1)), 0.04),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.04),
                (Brightness().only_on('img_pre'), 0.04),
                (Contrast().only_on('img_pre'), 0.04)), 0.1)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.04),
                (GaussianNoise().only_on('img_post'), 0.04),
                (Blur().only_on('img_post'), 0.1)), 0.04),
            (OneOf(
                (Saturation().only_on('img_post'), 0.04),
                (Brightness().only_on('img_post'), 0.04),
                (Contrast().only_on('img_post'), 0.04)), 0.1)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.04),
        Random(ElasticTransform().only_on('img_post'), p=0.04),
        Random(Dilation().only_on('msk'), p=0.9)
    )

    model = SeResnext50Classifier(SeResnext50Localizer.from_pretrained(version='0', seed=0).unet)
    optimizer = AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                           170, 180, 190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='0',
        seed=0,
        dataloader=DataLoader(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.1, 0.1, 0.3, 0.3, 0.2]), 0.5),
            (FocalLoss2d(class_weight=[0.1, 0.1, 0.3, 0.3, 0.2]), 2.),
            (SegmentationCCE(), 11)
        ),
        score=xview2.classification_score,
        epochs=20,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=0.999,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False,
            ),
            transform=None,
            interval=2,
        )
    )


def tune_classifier(seed: int):
    set_random_seeds(131313 + seed)
    transform = nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.01),
        Random(RotateAndScale(), p=0.5),
        BestCrop(samples=10, dsize=(512, 512), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.01),
            (RGBShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.01),
            (HSVShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.04),
                (GaussianNoise().only_on('img_pre'), 0.04),
                (Blur().only_on('img_pre'), 0.04)), 0.01),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.01),
                (Brightness().only_on('img_pre'), 0.01),
                (Contrast().only_on('img_pre'), 0.01)), 0.01)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.01),
                (GaussianNoise().only_on('img_post'), 0.01),
                (Blur().only_on('img_post'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img_post'), 0.01),
                (Brightness().only_on('img_post'), 0.01),
                (Contrast().only_on('img_post'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.017),
        Random(ElasticTransform().only_on('img_post'), p=0.017),
        Random(Dilation().only_on('msk'), p=0.9)
    )

    model = SeResnext50Classifier.from_checkpoint(version='0', seed=0)
    optimizer = AdamW(model.parameters(), lr=0.00001, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                           90, 110, 130, 150, 170, 180, 190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='tuned',
        seed=0,
        dataloader=DataLoader(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.1, 0.1, 0.3, 0.3, 0.2]), 0.5),
            (FocalLoss2d(class_weight=[0.1, 0.1, 0.3, 0.3, 0.2]), 2.),
            (SegmentationCCE(), 11)
        ),
        score=xview2.classification_score,
        epochs=2,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=0.999,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False,
            ),
            transform=None,
            interval=1
        )
    )
