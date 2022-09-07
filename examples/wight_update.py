# dpn92 unet double
self._optimizer.zero_grad()
self._grad_scaler.scale(loss).backward()
self._grad_scaler.unscale_(self._optimizer)
torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
self._grad_scaler.step(self._optimizer)
self._grad_scaler.update()

# resnet34 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# senet154 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# seresnext50 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# dpn92 unet localization
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# resnet34 unet localization
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# senet 154 unet localization
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# seresnext 50 unet localization
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# tune
# dpn92 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# resnet 34 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# se154 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# seresnext 50 unet double
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

# dpn92 unet loc
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()
# seresnext 50 unet loc
self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()