from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torch import nn

from metadamagenet.utils import set_random_seeds
from metadamagenet.dataset import LocalizationDataset, ClassificationDataset
from metadamagenet.models import Localizer, Classifier
from metadamagenet.models.unet import SeNet154Unet
from metadamagenet.losses import WeightedSum, BinaryFocalLoss2d, BinaryDiceLossWithLogits, DiceLoss, \
    SegmentationCCE
from metadamagenet.metrics import xview2
from metadamagenet.runner import Trainer, ValidationInTrainingParams
from metadamagenet.augment import Random, VFlip, Rotate90, Shift, RotateAndScale, BestCrop, OneOf, RGBShift, HSVShift, \
    Clahe, GaussianNoise, Blur, Saturation, Brightness, Contrast, ElasticTransform, Dilation

train_dir = Path('/datasets/xview2/train')
test_dir = Path('/datasets/xview2/test')


def train_localizer(seed: int):
    set_random_seeds(321 + seed)

    transform = nn.Sequential(
        Random(VFlip(), p=0.4),
        Random(Rotate90(), p=0.9),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.3),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.6),
        BestCrop(samples=5, dsize=(480, 480), size_range=(0.42, 0.52)),
        Random(RGBShift().only_on('img'), p=0.05),
        Random(HSVShift().only_on('img'), p=0.04),
        OneOf(
            (OneOf(
                (Clahe().only_on('img'), 0.08),
                (GaussianNoise().only_on('img'), 0.08),
                (Blur().only_on('img'), 0.08)), 0.08),
            (OneOf(
                (Saturation().only_on('img'), 0.08),
                (Brightness().only_on('img'), 0.08),
                (Contrast().only_on('img'), 0.08)), 0.08)
        ),
        Random(ElasticTransform(), p=0.05)
    )

    model = Localizer[SeNet154Unet](SeNet154Unet(pretrained_backbone=True))
    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[3, 7, 11, 15, 19, 23, 27, 33, 41, 50, 60, 70, 90, 110, 130, 150, 170, 180,
                                           190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='1',
        seed=0,
        dataloader=DataLoader(
            dataset=LocalizationDataset(train_dir),
            batch_size=14,
            num_workers=6,
            shuffle=True,
            drop_last=True
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (BinaryDiceLossWithLogits(), 1.),
            (BinaryFocalLoss2d(), 14.)
        ),
        score=xview2.localization_score,
        clip_grad_norm=0.999,
        epochs=30,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=6,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=1,
        )
    )


def train_classifier(seed: int):
    set_random_seeds(123123 + seed)
    # from dpn92-classifier-train
    transform = transform = nn.Sequential(
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
    model = Classifier[SeNet154Unet](Localizer[SeNet154Unet].from_pretrained(version='1', seed=seed).unet)
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[3, 5, 9, 13, 17, 21, 25, 29, 33, 47, 50, 60, 70, 90, 110,
                                           130, 150, 170, 180, 190],
                               gamma=0.5)
    Trainer(
        model=model,
        version='1',
        seed=0,
        dataloader=DataLoader(
            dataset=ClassificationDataset(train_dir),
            batch_size=8,
            num_workers=6,
            shuffle=True,
            drop_last=True,
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.1, 0.1, 0.6, 0.3, 0.2]), 1),
            (SegmentationCCE(), 8)
        ),
        score=xview2.classification_score,
        clip_grad_norm=0.999,
        epochs=16,
        grad_scaler=amp.GradScaler(),
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=2,
                num_workers=6,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=2,
        )
    )


def tune_classifier(seed: int):
    set_random_seeds(531 + seed)
    # from dpn92-classifier-tune
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

    model = Classifier[SeNet154Unet].from_pretrained(version='1', seed=0)
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
            batch_size=8,
            num_workers=6,
            shuffle=True,
            drop_last=True
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.1, 0.1, 0.6, 0.3, 0.2]), 1),
            (SegmentationCCE(), 8)
        ),
        score=xview2.classification_score,
        clip_grad_norm=0.999,
        epochs=2,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=2,
                num_workers=6,
                shuffle=False,
                drop_last=False
            ),
            transform=None,
            interval=2,
        )
    )
