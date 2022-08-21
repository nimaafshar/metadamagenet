import timeit

import cv2
import torch.nn

from src.configs import GeneralConfig
from src.zoo.models import SeResNext50_Unet_Loc, Dpn92_Unet_Loc, SeNet154_Unet_Loc, Res34_Unet_Loc
from src.setup import set_random_seeds
from src.model_config import ModelConfig
from src.file_structure import Dataset
from src.predict.loc import LocalizationPredictor
from src.logs import log

set_random_seeds()
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    GeneralConfig.load()
    config = GeneralConfig.get_instance()

    dataset = Dataset(config.test_dirs)

    model_configs = []

    for seed in [0, 1, 2]:
        model_configs.extend((
            ModelConfig(
                name="res50_loc",
                empty_model=SeResNext50_Unet_Loc().cuda(),
                version="0",
                seed=seed
            ),
            ModelConfig(
                name="dpn92_loc",
                empty_model=Dpn92_Unet_Loc().cuda(),
                version="0",
                seed=seed
            ),
            ModelConfig(
                name="se154_loc",
                empty_model=torch.nn.DataParallel(SeNet154_Unet_Loc().cuda()).cuda(),
                version="0",
                seed=seed
            ),
            ModelConfig(
                name="res34_loc",
                empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()).cuda(),
                version="1",
                seed=seed
            )
        ))

    LocalizationPredictor(model_configs,
                          config.predictions_dir,
                          dataset)

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
