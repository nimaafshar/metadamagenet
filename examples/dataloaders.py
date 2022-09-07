# dpn92 unet double
DataLoader(self._config.train_dataset,
           batch_size=12,
           num_workers=6,
           shuffle=True,
           pin_memory=False,
           drop_last=True)

DataLoader(self._config.validation_dataset,
           batch_size=4,
           num_workers=6,
           shuffle=False,
           pin_memory=False)

# resnet34 unet double
batch_size=16
num_workers=6


batch_size=8
num_workers=6

# senet154 unet double
batch_size=8,
num_workers=6,


batch_size=2,
num_workers=6,

# seresnext50 unet double

batch_size=16,
num_workers=6,

batch_size=4,
num_workers=6,

# dpn92 unet localization
batch_size=10,
num_workers=5,

batch_size=4,
num_workers=5,


# resnet34 unet localization
batch_size=16,
num_workers=6,

batch_size=8,
num_workers=6,

# senet 154 unet localization
batch_size=14,
num_workers=6,

batch_size=4,
num_workers=6,
# seresnext 50 unet localization

batch_size=15,
num_workers=5,

batch_size=4,
num_workers=5,

# tune
# dpn92 unet double
batch_size=12,
num_workers=6,

batch_size=4,
num_workers=6,

# resnet 34 unet double
batch_size=16,
num_workers=6,


batch_size=8,
num_workers=6,

# se154 unet double
batch_size=8,
num_workers=6,

batch_size=2,
num_workers=6,

# seresnext 50 unet double
batch_size=16,
num_workers=6,

batch_size=4,
num_workers=6,

# dpn92 unet loc
batch_size=10,
num_workers=6,

batch_size=4,
num_workers=6,

# seresnext 50 unet loc
batch_size=15,
num_workers=6,

batch_size=4,
num_workers=6,
