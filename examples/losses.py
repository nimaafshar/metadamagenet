from metadamagenet.losses import DiceLoss, WithSigmoid, FocalLoss2d, ComboLoss, ChanneledLoss

from torch import nn

# dpn92 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 0.5),
    (WithSigmoid(FocalLoss2d()), 5.0)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.1),
    (seg_loss, 0.1),
    (seg_loss, 0.5),
    (seg_loss, 0.3),
    (seg_loss, 0.2),
)

final_loss: ComboLoss = ComboLoss(
    (channeled, 1),
    (nn.CrossEntropyLoss(), 11)
).cuda()

# resnet34 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 12.0)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.05),
    (seg_loss, 0.2),
    (seg_loss, 0.8),
    (seg_loss, 0.7),
    (seg_loss, 0.4),
)

# senet154 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 0.5)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.1),
    (seg_loss, 0.1),
    (seg_loss, 0.6),
    (seg_loss, 0.3),
    (seg_loss, 0.2),
)

final_loss: ComboLoss = ComboLoss(
    (channeled, 1),
    (nn.CrossEntropyLoss(), 8)
)

# seresnext50 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 0.5),
    (WithSigmoid(FocalLoss2d()), 2.0)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.1),
    (seg_loss, 0.1),
    (seg_loss, 0.3),
    (seg_loss, 0.3),
    (seg_loss, 0.2),
)

final_loss: ComboLoss = ComboLoss(
    (channeled, 1),
    (nn.CrossEntropyLoss(), 11)
)

# dpn92 unet localization
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 6.0)
)


# resnet34 unet localization
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 10.0)
)

# senet 154 unet localization
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 14.0)
)

# seresnext 50 unet localization
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 10.0)
)


# tune
# dpn92 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 0.5),
    (WithSigmoid(FocalLoss2d()), 5.0)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.1),
    (seg_loss, 0.1),
    (seg_loss, 0.5),
    (seg_loss, 0.3),
    (seg_loss, 0.2),
)

final_loss: ComboLoss = ComboLoss(
    (channeled, 1),
    (nn.CrossEntropyLoss(), 11)
)

# resnet 34 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 12.0)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.05),
    (seg_loss, 0.2),
    (seg_loss, 0.8),
    (seg_loss, 0.7),
    (seg_loss, 0.4),
)

# se154 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 0.5)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.1),
    (seg_loss, 0.1),
    (seg_loss, 0.6),
    (seg_loss, 0.3),
    (seg_loss, 0.2),
)

final_loss: ComboLoss = ComboLoss(
    (channeled, 1),
    (nn.CrossEntropyLoss(), 8)
)

# seresnext 50 unet double
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 0.5),
    (WithSigmoid(FocalLoss2d()), 2.0)
)

channeled: ChanneledLoss = ChanneledLoss(
    (seg_loss, 0.1),
    (seg_loss, 0.1),
    (seg_loss, 0.3),
    (seg_loss, 0.3),
    (seg_loss, 0.2),
)

final_loss: ComboLoss = ComboLoss(
    (channeled, 1),
    (nn.CrossEntropyLoss(), 11)
)

# dpn92 unet loc
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 6.0)
)

# seresnext 50 unet loc
seg_loss: ComboLoss = ComboLoss(
    (WithSigmoid(DiceLoss()), 1.0),
    (WithSigmoid(FocalLoss2d()), 10.0)
)
