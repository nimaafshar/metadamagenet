# dpn92 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                    170, 180, 190],
                                        gamma=0.5)

# resnet34 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                    170, 180, 190],
                                        gamma=0.5)

# senet154 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[3, 5, 9, 13, 17, 21, 25, 29, 33, 47, 50, 60, 70, 90, 110,
                                                    130, 150, 170, 180, 190],
                                        gamma=0.5)
# seresnext50 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                    170, 180, 190],
                                        gamma=0.5)

# dpn92 unet localization
lr_scheduler = MultiStepLR(optimizer,
                           milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180, 190],
                           gamma=0.5)

# resnet34 unet localization
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[5, 11, 17, 25, 33, 47, 50,
                                                    60, 70, 90, 110, 130, 150,
                                                    170, 180, 190],
                                        gamma=0.5)

# senet 154 unet localization
lr_scheduler = MultiStepLR(optimizer,
                           milestones=[3, 7, 11, 15, 19, 23, 27, 33, 41, 50, 60, 70, 90, 110, 130, 150, 170,
                                       180, 190],
                           gamma=0.5)

# seresnext 50 unet localization
lr_scheduler = MultiStepLR(optimizer,
                           milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180, 190],
                           gamma=0.5)

# tune
# dpn92 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90,
                                                    110, 130, 150, 170, 180, 190],
                                        gamma=0.5)
# resnet 34 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90,
                                                    110, 130, 150, 170, 180, 190],
                                        gamma=0.5)

# se154 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90,
                                                    110, 130, 150, 170, 180, 190],
                                        gamma=0.5)

# seresnext 50 unet double
lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                        milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90,
                                                    110, 130, 150, 170, 180, 190],
                                        gamma=0.5)

# dpn92 unet loc
lr_scheduler = MultiStepLR(optimizer,
                           milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130,
                                       150, 170, 180, 190],
                           gamma=0.5)

# seresnext 50 unet loc
lr_scheduler = MultiStepLR(optimizer,
                           milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130,
                                       150, 170, 180, 190],
                           gamma=0.5)
