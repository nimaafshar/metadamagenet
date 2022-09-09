cudnn.benchmark = True

# dpn92 unet double
torch.softmax(model_out, dim=1)
# resnet34 unet double
torch.sigmoid(model_out)
# senet154 unet double
torch.softmax(model_out, dim=1)
# seresnext50 unet double
torch.softmax(model_out, dim=1)
#dpn92 unet localization


#resnet34 unet localization


#senet 154 unet localization



#seresnext 50 unet localization



# tune
#dpn92 unet double
torch.softmax(model_out, dim=1)
#resnet 34 unet double
torch.sigmoid(model_out)
#se154 unet double
torch.softmax(model_out, dim=1)
#seresnext 50 unet double
torch.softmax(model_out, dim=1)
#dpn92 unet loc

#seresnext 50 unet loc
