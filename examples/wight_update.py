# dpn92 unet double
self._optimizer.zero_grad()
self._grad_scaler.scale(loss).backward()
self._grad_scaler.unscale_(self._optimizer)
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
self._grad_scaler.step(self._optimizer)
self._grad_scaler.update()

# resnet34 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# senet154 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# seresnext50 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# dpn92 unet localization
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)

# resnet34 unet localization
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# senet 154 unet localization
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# seresnext 50 unet localization
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)

# tune
# dpn92 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# resnet 34 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# se154 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# seresnext 50 unet double
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)

# dpn92 unet loc
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)

# seresnext 50 unet loc
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)
