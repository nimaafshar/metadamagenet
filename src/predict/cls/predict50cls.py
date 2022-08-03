import sys
import timeit
import cv2

from src.configs import TEST_DIR
from src.setup import set_random_seeds
from src.zoo.models import SeResNext50_Unet_Double
from src.file_structure import Dataset
from src.logs import log
from src.model_config import ModelConfig
from src.predict.cls import SoftmaxClassificationPredictor

set_random_seeds()
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    model_config = ModelConfig(
        name='res50_cls_cce',
        model_type=SeResNext50_Unet_Double,
        seed=seed,
        tuned=True
    )

    test_dataset = Dataset((TEST_DIR,))
    predictor: SoftmaxClassificationPredictor = SoftmaxClassificationPredictor(model_config, test_dataset)
    predictor.predict()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
