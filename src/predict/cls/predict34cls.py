from os import path, makedirs, listdir
import sys

import emoji
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.autograd import Variable

from tqdm import tqdm
import timeit
import cv2

from src.configs import MODELS_WEIGHTS_FOLDER, TEST_DIR
from src.setup import set_random_seeds
from src.util.utils import normalize_colors
from src.zoo.models import Res34_Unet_Double
from src.file_structure import Dataset, ImageData, DataTime
from src.logs import log
from src.model_config import ModelConfig

set_random_seeds()
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def predict(seed: int):
    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev
    # cudnn.benchmark = True

    model_config = ModelConfig(
        name='res34cls2',
        model_type=Res34_Unet_Double,
        seed=seed,
        tuned=True
    )

    model_config.pred_directory.mkdir(parents=False, exist_ok=True)
    log(f':file_folder: directory {model_config.pred_directory} created to save predictions')

    model = model_config.model_type().cuda()
    model = nn.DataParallel(model).cuda()

    log(f":arrow_up: loading checkpoint '{model_config.best_snap_path}'")
    checkpoint = torch.load(model_config.best_snap_path, map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()

    # loading parts of the state dict that are saved in the checkpoint
    for k in sd:
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    log(f":white_check_mark: loaded checkpoint '{model_config.best_snap_path}' "
        f"[epoch={checkpoint['epoch']}, best_score={checkpoint['best_score']}]")

    model.eval()

    test_dataset = Dataset((TEST_DIR,))

    log(f":mag: discovering {TEST_DIR.absolute()}...")
    test_dataset.discover()
    log(f":file_folder: {len(test_dataset)} files found")

    with torch.no_grad():
        image_data: ImageData
        for image_data in tqdm(test_dataset.images):
            pre_image: npt.NDArray = cv2.imread(image_data.image(DataTime.PRE), cv2.IMREAD_COLOR)
            post_image: npt.NDArray = cv2.imread(image_data.image(DataTime.POST), cv2.IMREAD_COLOR)

            img = np.concatenate((pre_image, post_image), axis=2)
            img = normalize_colors(img)

            # test-time augmentations
            inp = np.asarray((img,  # original
                              img[::-1, ...],  # flip up-down
                              img[:, ::-1, ...],  # flip left-right
                              img[::-1, ::-1, ...]), dtype='float')  # flip along both x and y-axis (180 rotation)
            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = inp.cuda()

            msk = model(inp)
            msk = torch.sigmoid(msk)
            msk = msk.cpu().numpy()

            pred = np.asarray((msk[0, ...],
                               msk[1, :, ::-1, :],  # flip left-right
                               msk[2, :, :, ::-1],  # flip from BRG to RGB
                               msk[3, :, ::-1, ::-1])).mean(axis=0)  # left-right and RGB to BRG flip

            msk = pred * 255
            msk = msk.astype('uint8').transpose(1, 2, 0)

            # write predictions to file

            cv2.imwrite(model_config.pred_directory / f'{image_data.name(DataTime.PRE)}_part1.png',
                        msk[..., :3],
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

            cv2.imwrite(model_config.pred_directory / f'{image_data.name(DataTime.PRE)}_part2.png',
                        msk[..., 2:],
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # FIXME: what is part1 and part2?


if __name__ == '__main__':
    t0 = timeit.default_timer()
    predict(int(sys.argv[1]))

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
