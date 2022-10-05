from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torch import nn

from metadamagenet.utils import set_random_seeds
from metadamagenet.dataset import LocalizationDataset, ClassificationDataset
from metadamagenet.models import Localizer, Classifier
from metadamagenet.models.unet import Resnet34Unet
from metadamagenet.losses import WeightedSum, BinaryFocalLoss2d, BinaryDiceLossWithLogits, DiceLoss, FocalLoss2d
from metadamagenet.metrics import xview2
from metadamagenet.runner import Trainer, ValidationInTrainingParams
from metadamagenet.augment import Random, VFlip, Rotate90, Shift, RotateAndScale, BestCrop, OneOf, RGBShift, HSVShift, \
    Clahe, GaussianNoise, Blur, Saturation, Brightness, Contrast, ElasticTransform, Dilation

train_dir = Path('/datasets/xview2/train')
test_dir = Path('/datasets/xview2/test')


def train_localizer(seed: int):
    set_random_seeds(545 + seed)

    transform = nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.2),
        Random(RotateAndScale(), p=0.8),
        BestCrop(samples=5, dsize=(736, 736), size_range=(0.6, 0.9)),
        OneOf(
            (RGBShift().only_on('img'), 0.03),
            (HSVShift().only_on('img'), 0.03)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img'), 0.03),
                (GaussianNoise().only_on('img'), 0.03),
                (Blur().only_on('img'), 0.02)), 0.07),
            (OneOf(
                (Saturation().only_on('img'), 0.03),
                (Brightness().only_on('img'), 0.03),
                (Contrast().only_on('img'), 0.03)), 0.07)
        ),
        Random(ElasticTransform().only_on('img'), p=0.03)
    )
    Trainer(
        model=Localizer[Resnet34Unet](Resnet34Unet(pretrained_backbone=True)),
        version='1',
        seed=seed,
        dataloader=DataLoader(
            dataset=LocalizationDataset(train_dir),
            shuffle=True,
            batch_size=16,
            num_workers=6,
            drop_last=True
        ),
        transform=transform,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[5, 11, 17, 25, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                               170, 180, 190],
                                                   gamma=0.5),
        loss=WeightedSum(
            (BinaryDiceLossWithLogits(), 1.0),
            (BinaryFocalLoss2d(), 10.0)
        ),
        score=xview2.localization_score,
        epochs=55,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=0.999,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=LocalizationDataset(test_dir),
                batch_size=8,
                num_workers=6
            ),
            interval=2,
        )
    ).run()


def train_classifier(seed: int):
    set_random_seeds(321 + seed)

    transform = nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.1),
        Random(RotateAndScale(), p=0.4),
        BestCrop(samples=10, dsize=(608, 608), size_range=(0.65, 0.85)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.015),
            (RGBShift().only_on('img_post'), 0.015),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.015),
            (HSVShift().only_on('img_post'), 0.015),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.015),
                (GaussianNoise().only_on('img_pre'), 0.015),
                (Blur().only_on('img_pre'), 0.1)), 0.015),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.015),
                (Brightness().only_on('img_pre'), 0.015),
                (Contrast().only_on('img_pre'), 0.015)), 0.02)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.015),
                (GaussianNoise().only_on('img_post'), 0.015),
                (Blur().only_on('img_post'), 0.1)), 0.015),
            (OneOf(
                (Saturation().only_on('img_post'), 0.015),
                (Brightness().only_on('img_post'), 0.015),
                (Contrast().only_on('img_post'), 0.015)), 0.02)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.017),
        Random(ElasticTransform().only_on('img_post'), p=0.017),
        Random(Dilation().only_on('msk'), p=0.9)
    )

    model = Classifier[Resnet34Unet](Localizer[Resnet34Unet].from_pretrained(version='1', seed=0).unet)
    opt = AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6)
    lrs = MultiStepLR(opt,
                      milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                  170, 180, 190],
                      gamma=0.5)
    Trainer(
        model=model,
        version='1',
        seed=0,
        dataloader=DataLoader(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            num_workers=6,
            shuffle=True,
            drop_last=True
        ),
        transform=transform,
        optimizer=opt,
        lr_scheduler=lrs,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.05, 0.2, 0.8, 0.7, 0.4]), 1.),
            (FocalLoss2d(class_weight=[0.05, 0.2, 0.8, 0.7, 0.4]), 12.0)
        ),
        score=xview2.classification_score,
        epochs=20,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=0.999,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                batch_size=8,
                shuffle=False,
                drop_last=False,
                num_workers=6
            ),
            transform=None,
            interval=2
        )
    )


def tune_classifier(seed: int):
    set_random_seeds(seed + 357)

    transform = nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.02),
        Random(RotateAndScale(), p=0.5),
        BestCrop(samples=10, dsize=(608, 608), size_range=(0.65, 0.85)),
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
                (Clahe().only_on('img_post'), 0.015),
                (GaussianNoise().only_on('img_post'), 0.015),
                (Blur().only_on('img_post'), 0.015)), 0.01),
            (OneOf(
                (Saturation().only_on('img_post'), 0.01),
                (Brightness().only_on('img_post'), 0.01),
                (Contrast().only_on('img_post'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.01),
        Random(ElasticTransform().only_on('img_post'), p=0.01),
        Random(Dilation().only_on('msk'), p=0.9)
    )

    model = Classifier[Resnet34Unet].from_checkpoint(version='1', seed=0)
    optimizer = AdamW(model.parameters(), lr=0.000008, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                           90, 110, 130, 150, 170, 180, 190],
                               gamma=0.5)
    Trainer(
        model=model,
        version="tuned",
        seed=0,
        dataloader=DataLoader(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            shuffle=True,
            num_workers=6,
            drop_last=True
        ),
        transform=transform,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=WeightedSum(
            (DiceLoss(class_weights=[0.05, 0.2, 0.8, 0.7, 0.4]), 1.),
            (FocalLoss2d(class_weight=[0.05, 0.2, 0.8, 0.7, 0.4]), 12.0)
        ),
        score=xview2.classification_score,
        epochs=3,
        grad_scaler=amp.GradScaler(),
        clip_grad_norm=0.999,
        validation_params=ValidationInTrainingParams(
            dataloader=DataLoader(
                dataset=ClassificationDataset(test_dir),
                shuffle=False,
                drop_last=False,
                batch_size=8,
                num_workers=6
            ),
            interval=1,
        )
    )
