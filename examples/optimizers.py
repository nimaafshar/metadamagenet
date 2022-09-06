# dpn92 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                 lr=0.0002,
                                 weight_decay=1e-6)

# resnet34 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                 lr=0.0002,
                                 weight_decay=1e-6)


# senet154 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.0001,
                                     weight_decay=1e-6)


# seresnext50 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.0002,
                                     weight_decay=1e-6)


# dpn92 unet localization
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)


# resnet34 unet localization
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)

# senet 154 unet localization
optimizer: Optimizer = AdamW(model.parameters(),
                             lr=0.00015,
                             weight_decay=1e-6)

# seresnext 50 unet localization
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)

# tune
# dpn92 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.000008,
                                     weight_decay=1e-6)


# resnet 34 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.000008,
                                     weight_decay=1e-6)


# se154 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.000008,
                                     weight_decay=1e-6)


# seresnext 50 unet double
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00001,
                                     weight_decay=1e-6)


# dpn92 unet loc
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00004,
                                     weight_decay=1e-6)


# seresnext 50 unet loc
optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00004,
                                     weight_decay=1e-6)

