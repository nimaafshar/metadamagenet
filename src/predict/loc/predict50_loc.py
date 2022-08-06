import timeit

import cv2

from src import configs
from src.zoo.models import SeResNext50_Unet_Loc
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

    model_configs = (
        ModelConfig(
            name="res50_loc",
            model_type=SeResNext50_Unet_Loc,
            version="tuned",
            seed=0
        ),
        ModelConfig(
            name="res50_loc",
            model_type=SeResNext50_Unet_Loc,
            version="tuned",
            seed=1
        ),
        ModelConfig(
            name="res50_loc",
            model_type=SeResNext50_Unet_Loc,
            version="tuned",
            seed=2
        )
    )

    LocalizationPredictor(model_configs,
                          configs.PREDICTIONS_DIRECTORY / 'res50_loc_tuned',
                          Dataset((configs.TEST_DIR,))
                          )

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
