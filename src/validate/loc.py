import torch
from tqdm import tqdm
from torch.backends import cudnn

from .validator import Validator
from src.util.utils import AverageMeter
from src.logs import log
from src.losses import dice_batch


class LocalizationValidator(Validator):
    def _setup(self):
        super(LocalizationValidator, self)._setup()
        # vis_dev = sys.argv[2]
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev
        cudnn.benchmark = True

    def _evaluate(self) -> float:
        meter = AverageMeter()
        self._model.eval()

        iterator = tqdm(self._dataloader)

        with torch.no_grad():
            for i, (img_batch, msk_batch) in enumerate(iterator):
                msk_batch: torch.FloatTensor = msk_batch.cuda(non_blocking=True)
                img_batch: torch.BoolTensor = img_batch.cuda(non_blocking=True)

                msk_pred: torch.FloatTensor

                if self._test_time_augmentor is not None:
                    augmented_batch: torch.FloatTensor = self._test_time_augmentor.augment(img_batch)
                    augmented_out_batch: torch.FloatTensor = self._model(augmented_batch)
                    augmented_msk_pred: torch.FloatTensor = torch.sigmoid(augmented_out_batch)
                    msk_pred = self._test_time_augmentor.aggregate(augmented_msk_pred)
                else:
                    out_batch: torch.FloatTensor = self._model(img_batch)
                    msk_pred: torch.FloatTensor = torch.sigmoid(out_batch)

                dice_scores = dice_batch(msk_batch[:, 0, ...], msk_pred[:, 0, ...] > self._evaluation_dice_thr)
                meter.update(float(dice_scores.mean()), n=img_batch.size(0))

                iterator.set_description(f" Dice {meter.val:.6f} ({meter.avg:.6f})")

        log(f"Validation set Dice: {meter.avg:.6f}")
        return meter.avg
