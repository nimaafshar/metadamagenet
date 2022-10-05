from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torch import nn

from metadamagenet.utils import set_random_seeds
from metadamagenet.dataset import LocalizationDataset, ClassificationDataset
from metadamagenet.models import Localizer, Classifier
from metadamagenet.models.unet import Dpn92Unet
from metadamagenet.losses import WeightedSum, BinaryFocalLoss2d, BinaryDiceLossWithLogits, DiceLoss, FocalLoss2d, \
    SegmentationCCE
from metadamagenet.metrics import xview2
from metadamagenet.runner import Trainer, ValidationInTrainingParams
from metadamagenet.augment import Random, VFlip, Rotate90, Shift, RotateAndScale, BestCrop, OneOf, RGBShift, HSVShift, \
    Clahe, GaussianNoise, Blur, Saturation, Brightness, Contrast, ElasticTransform, Dilation

train_dir = Path('/datasets/xview2/train')
test_dir = Path('/datasets/xview2/test')


def train_localizer(seed: int):
    set_random_seeds(111 + seed)

    transform = nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.1),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.1),
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
        Random(ElasticTransform(), p=0.001)
    )

    model = Localizer[Dpn92Unet](Dpn92Unet(pretrained_backbone=True))
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
            batch_size=10,
            num_workers=5,
            shuffle=True,
            drop_last=True
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (BinaryDiceLossWithLogits(), 1.0),
            (BinaryFocalLoss2d(), 6.0)
        ),
        score=xview2.localization_score,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=1.1,
        epochs=100,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=5,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=2
        )
    )


def tune_localizer(seed: int):
    set_random_seeds(156 + seed)

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

    model = Localizer[Dpn92Unet].from_checkpoint(version='0', seed=0)
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
            batch_size=10,
            num_workers=6,
            shuffle=True,
            drop_last=True
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (BinaryDiceLossWithLogits(), 1.0),
            (BinaryFocalLoss2d(), 6.0)
        ),
        score=xview2.localization_score,
        epochs=8,
        clip_grad_norm=1.1,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=2
        )
    )


def train_classifier(seed: int):
    set_random_seeds(54321 + seed)

    transform = nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.9999),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.5),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.95),
        BestCrop(samples=10, dsize=(512, 512), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.1),
            (RGBShift().only_on('img_post'), 0.1),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.1),
            (HSVShift().only_on('img_post'), 0.1),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.1),
                (GaussianNoise().only_on('img_pre'), 0.1),
                (Blur().only_on('img_pre'), 0.1)), 0.1),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.1),
                (Brightness().only_on('img_pre'), 0.1),
                (Contrast().only_on('img_pre'), 0.1)), 0.1)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.1),
                (GaussianNoise().only_on('img_post'), 0.1),
                (Blur().only_on('img_post'), 0.1)), 0.1),
            (OneOf(
                (Saturation().only_on('img_post'), 0.1),
                (Brightness().only_on('img_post'), 0.1),
                (Contrast().only_on('img_post'), 0.1)), 0.1)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.1),
        Random(ElasticTransform().only_on('img_post'), p=0.1),
        Random(Dilation().only_on('msk'), p=0.9)
    )

    model = Classifier[Dpn92Unet](Localizer[Dpn92Unet].from_pretrained(version='0', seed=0).unet)
    optimizer = AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                           170, 180, 190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='1',
        seed=0,
        dataloader=DataLoader(
            dataset=ClassificationDataset(train_dir),
            batch_size=12,
            num_workers=6,
            shuffle=True,
            drop_last=True
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.1, 0.1, 0.5, 0.3, 0.2]), 0.5),
            (FocalLoss2d(class_weight=[0.1, 0.1, 0.5, 0.3, 0.2]), 5.0),
            (SegmentationCCE(), 11)
        ),
        score=xview2.classification_score,
        epochs=10,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=0.999,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=2,
        )
    )


def tune_classifier(seed: int):
    set_random_seeds(seed + 777)

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
                (Clahe().only_on('img_pre'), 0.01),
                (GaussianNoise().only_on('img_pre'), 0.01),
                (Blur().only_on('img_pre'), 0.01)), 0.01),
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
        Random(ElasticTransform().only_on('img_pre'), p=0.01),
        Random(ElasticTransform().only_on('img_post'), p=0.01),
        Random(Dilation().only_on('msk'), p=0.9)
    )

    model = Classifier[Dpn92Unet].from_pretrained(version='1', seed=0)
    optimizer = AdamW(model.parameters(), lr=0.000008, weight_decay=1e-6)
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
            batch_size=12,
            num_workers=6,
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.1, 0.1, 0.5, 0.3, 0.2]), 0.5),
            (FocalLoss2d(class_weight=[0.1, 0.1, 0.5, 0.3, 0.2]), 5.0),
            (SegmentationCCE(), 11)
        ),
        score=xview2.classification_score,
        clip_grad_norm=0.999,
        epochs=1,
        grad_scaler=amp.GradScaler(),
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=1
        )
    )
