# dpn92 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 0.5, 'focal': 5.0}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()
label_loss_weights = np.array([0.1, 0.1, 0.5, 0.3, 0.2, 11])

# resnet34 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 1.0, 'focal': 12.0}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()

label_loss_weights = np.array([0.05, 0.2, 0.8, 0.7, 0.4])

# senet154 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 0.5}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()
label_loss_weights = np.array([0.1, 0.1, 0.6, 0.3, 0.2, 8])

# seresnext50 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()

#dpn92 unet localization
seg_loss = ComboLoss({'dice': 1.0, 'focal': 6.0}, per_image=False).cuda()

#resnet34 unet localization
seg_loss: ComboLoss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()

#senet 154 unet localization
seg_loss = ComboLoss({'dice': 1.0, 'focal': 14.0}, per_image=False).cuda()


#seresnext 50 unet localization
seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()


# tune
#dpn92 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 0.5, 'focal': 5.0}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()

#resnet 34 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 1.0, 'focal': 12.0}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()

#se154 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 0.5}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()

#seresnext 50 unet double
seg_loss: ComboLoss = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).cuda()
ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()

#dpn92 unet loc
seg_loss = ComboLoss({'dice': 1.0, 'focal': 6.0}, per_image=False).cuda()

#seresnext 50 unet loc
seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()