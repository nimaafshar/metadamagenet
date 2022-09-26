from metadamagenet.losses import DiceLoss, WithSigmoid, FocalLoss2d, WeightedLoss, ChanneledLoss, SegCCE
from enum import Enum


class Losses(Enum):
    DPN92_LOC = WeightedLoss(
        ("Dice", WithSigmoid(DiceLoss()), 1.0),
        ("Focal", WithSigmoid(FocalLoss2d()), 6.0)
    )
    DPN92_CLS = WeightedLoss(
        (ChanneledLoss(
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 5.0)), 0.1),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 5.0)), 0.1),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 5.0)), 0.5),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 5.0)), 0.3),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 5.0)), 0.2)), 1),
        (SegCCE(), 11)
    )

    RESNET34_LOC = WeightedLoss(
        ("Dice", WithSigmoid(DiceLoss()), 1.0),
        ("Focal", WithSigmoid(FocalLoss2d()), 10.0)
    )

    RESNET34_CLS = ChanneledLoss(
        (WeightedLoss(
            ("Dice", WithSigmoid(DiceLoss()), 1.0),
            ("Focal", WithSigmoid(FocalLoss2d()), 12.0)), 0.05),
        (WeightedLoss(
            ("Dice", WithSigmoid(DiceLoss()), 1.0),
            ("Focal", WithSigmoid(FocalLoss2d()), 12.0)), 0.2),
        (WeightedLoss(
            ("Dice", WithSigmoid(DiceLoss()), 1.0),
            ("Focal", WithSigmoid(FocalLoss2d()), 12.0)), 0.8),
        (WeightedLoss(
            ("Dice", WithSigmoid(DiceLoss()), 1.0),
            ("Focal", WithSigmoid(FocalLoss2d()), 12.0)), 0.7),
        (WeightedLoss(
            ("Dice", WithSigmoid(DiceLoss()), 1.0),
            ("Focal", WithSigmoid(FocalLoss2d()), 12.0)), 0.4),
    )

    SENET154_LOC = WeightedLoss(
        ("Dice", WithSigmoid(DiceLoss()), 1.0),
        ("Focal", WithSigmoid(FocalLoss2d()), 14.0)
    )

    SENET154_CLS = WeightedLoss(
        (ChanneledLoss(
            (WithSigmoid(DiceLoss()), 0.1),
            (WithSigmoid(DiceLoss()), 0.1),
            (WithSigmoid(DiceLoss()), 0.6),
            (WithSigmoid(DiceLoss()), 0.3),
            (WithSigmoid(DiceLoss()), 0.2)), 1),
        (SegCCE(), 8)
    )

    SERESNEXT50_LOC = WeightedLoss(
        ("Dice", WithSigmoid(DiceLoss()), 1.0),
        ("Focal", WithSigmoid(FocalLoss2d()), 10.0)
    )

    SERESNEXT50_CLS = WeightedLoss(
        (ChanneledLoss(
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 2.0)), 0.1),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 2.0)), 0.1),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 2.0)), 0.3),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 2.0)), 0.3),
            (WeightedLoss(
                ("Dice", WithSigmoid(DiceLoss()), 0.5),
                ("Focal", WithSigmoid(FocalLoss2d()), 2.0)), 0.2)), 1),
        (SegCCE(), 11)
    )
