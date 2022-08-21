import timeit
import cv2
import torch.nn

from src.configs import GeneralConfig
from src.zoo.models import Res34_Unet_Loc
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

    model_configs = (
        ModelConfig(
            name="res34_loc",
            empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()).cuda(),
            version='1',
            seed=0
        ),
        ModelConfig(
            name="res34_loc",
            empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()).cuda(),
            version='1',
            seed=1
        ),
        ModelConfig(
            name="res34_loc",
            empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()).cuda(),
            version='1',
            seed=2
        )
    )

    predictor: LocalizationPredictor = LocalizationPredictor(model_configs,
                                                             config.predictions_dir / 'res34_loc/',
                                                             Dataset(config.test_dirs))
    predictor.predict()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
