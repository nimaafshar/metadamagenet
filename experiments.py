import sys
from typing import Dict
from pathlib import Path

from torch.optim import AdamW

from torch.optim.lr_scheduler import MultiStepLR

from metadamagenet.cmd import Data, ValidateInTraining, Train, init_classifier, Command
from metadamagenet.dataset import LocalizationDataset, ClassificationDataset, \
    LocalizationPreprocessor, ClassificationPreprocessor
from metadamagenet.wrapper import Resnet34LocalizerWrapper, Resnet34ClassifierWrapper, \
    Dpn92ClassifierWrapper, Dpn92LocalizerWrapper, SeResnext50LocalizerWrapper, SeResnext50ClassifierWrapper, \
    SeNet154LocalizerWrapper, SeNet154ClassifierWrapper

from example_losses import Losses
from example_preprocessors import Preprocessors

train_dir = Path("/datasets/xview2/train")
test_dir = Path("/datasets/xview2/test")

configs: Dict[str, Command] = {
    "resnet34_loc_train": Train(
        model=lambda: Resnet34LocalizerWrapper().from_backbone(),
        version='1',
        seed=0,
        wrapper=Resnet34LocalizerWrapper(),
        data=Data(
            dataset=LocalizationDataset(train_dir),
            batch_size=16,
            num_workers=6
        ),
        preprocessor=Preprocessors.RESNET34_LOC,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[5, 11, 17, 25, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                               170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.RESNET34_LOC,
        score=Resnet34LocalizerWrapper.default_score,
        epochs=55,
        random_seed=545,
        grad_scaling=True,
        clip_grad_norm=0.999,
        validation=ValidateInTraining(
            data=Data(
                dataset=LocalizationDataset(test_dir),
                batch_size=8,
                num_workers=6
            ),
            preprocessor=LocalizationPreprocessor(),
            interval=2,
        )
    ),
    "resnet34_cls_train": Train(
        model=init_classifier(Resnet34ClassifierWrapper, Resnet34LocalizerWrapper, loc_version='1', loc_seed='0'),
        version='1',
        wrapper=Resnet34ClassifierWrapper(),
        seed=0,
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            num_workers=6
        ),
        preprocessor=Preprocessors.RESNET34_CLS,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                               170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.RESNET34_CLS,
        score=Resnet34ClassifierWrapper.default_score,
        epochs=20,
        random_seed=321,
        grad_scaling=True,
        clip_grad_norm=0.999,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=8,
                num_workers=6
            ),
            preprocessor=ClassificationPreprocessor(),
            interval=2
        )
    ),
    "resnet34_cls_tune": Train(
        model=lambda: Resnet34LocalizerWrapper().from_checkpoint(version='1', seed=0),
        version="tuned",
        seed=0,
        wrapper=Resnet34ClassifierWrapper(),
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            num_workers=6
        ),
        preprocessor=Preprocessors.RESNET34_CLS_TUNE,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.000008, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                                               90, 110, 130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.RESNET34_CLS,
        score=Resnet34ClassifierWrapper.default_score,
        epochs=3,
        random_seed=357,
        grad_scaling=True,
        clip_grad_norm=0.999,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=8,
                num_workers=6
            ),
            preprocessor=ClassificationPreprocessor(),
            interval=1,
        )
    ),
    "seresnext50_loc_train": Train(
        model=lambda: SeResnext50LocalizerWrapper().from_backbone(),
        version='0',
        seed=0,
        wrapper=SeResnext50LocalizerWrapper(),
        data=Data(
            dataset=LocalizationDataset(train_dir),
            batch_size=15,
            num_workers=5,
        ),
        preprocessor=Preprocessors.SERESNEXT50_LOC,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180,
                                                               190],
                                                   gamma=0.5),
        loss=Losses.SERESNEXT50_LOC,
        score=SeResnext50LocalizerWrapper.default_score,
        epochs=150,
        random_seed=123,
        grad_scaling=True,
        clip_grad_norm=1.1,
        validation=ValidateInTraining(
            data=Data(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=5,
            ),
            preprocessor=LocalizationPreprocessor(),
            interval=2,
        )
    ),
    "seresnext50_loc_tune": Train(
        model=lambda: SeResnext50LocalizerWrapper().from_checkpoint(version='0', seed=0),
        version='0',
        seed=0,
        wrapper=SeResnext50LocalizerWrapper(),
        data=Data(
            dataset=LocalizationDataset(train_dir),
            batch_size=15,
            num_workers=6,
        ),
        preprocessor=Preprocessors.DPN92_CLS_TUNE,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00004, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                                               90, 110, 130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.SERESNEXT50_LOC,
        score=SeResnext50LocalizerWrapper.default_score,
        epochs=12,
        random_seed=432,
        grad_scaling=True,
        clip_grad_norm=1.1,
        validation=ValidateInTraining(
            data=Data(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=6,
            ),
            preprocessor=LocalizationPreprocessor(),
            interval=1,
        )
    ),
    "seresnext50_cls_train": Train(
        model=init_classifier(SeResnext50ClassifierWrapper, SeResnext50LocalizerWrapper, loc_version='0', loc_seed=0),
        version='0',
        seed=0,
        wrapper=SeResnext50ClassifierWrapper(),
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            num_workers=6,
        ),
        preprocessor=Preprocessors.SERESNEXT50_CLS,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                               170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.SERESNEXT50_CLS,
        score=SeResnext50ClassifierWrapper.default_score,
        epochs=20,
        random_seed=1234,
        grad_scaling=True,
        clip_grad_norm=0.999,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6
            ),
            preprocessor=ClassificationPreprocessor(),
            interval=2,
        )
    ),
    "seresnext50_cls_tune": Train(
        model=lambda: SeResnext50ClassifierWrapper().from_checkpoint(version='0', seed=0),
        version='tuned',
        seed=0,
        wrapper=SeResnext50ClassifierWrapper(),
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=16,
            num_workers=6,
        ),
        preprocessor=Preprocessors.SERESNEXT50_CLS_TUNE,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00001, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                                               90, 110, 130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.SERESNEXT50_CLS,
        score=SeResnext50ClassifierWrapper.default_score,
        epochs=2,
        random_seed=131313,
        grad_scaling=True,
        clip_grad_norm=0.999,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6
            ),
            preprocessor=ClassificationPreprocessor(),
        )
    ),
    "dpn92_loc_train": Train(
        model=lambda: Dpn92LocalizerWrapper().from_backbone(),
        version='0',
        seed=0,
        wrapper=Dpn92LocalizerWrapper(),
        data=Data(
            dataset=LocalizationDataset(train_dir),
            batch_size=10,
            num_workers=5,
        ),
        preprocessor=Preprocessors.DPN92_LOC,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180,
                                                               190],
                                                   gamma=0.5),
        loss=Losses.DPN92_LOC,
        score=Dpn92LocalizerWrapper.default_score,
        grad_scaling=True,
        clip_grad_norm=1.1,
        random_seed=111,
        epochs=100,
        validation=ValidateInTraining(
            data=Data(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=5,
            ),
            preprocessor=LocalizationPreprocessor(),
            interval=2
        )
    ),
    "dpn92_loc_tune": Train(
        model=lambda: Dpn92LocalizerWrapper().from_checkpoint(version='0', seed=0),
        version='0',
        seed=0,
        wrapper=Dpn92LocalizerWrapper(),
        data=Data(
            dataset=LocalizationDataset(train_dir),
            batch_size=10,
            num_workers=6,
        ),
        preprocessor=Preprocessors.DPN92_LOC_TUNE,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00004, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                                               90, 110, 130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.DPN92_LOC,
        score=Dpn92LocalizerWrapper.default_score,
        epochs=8,
        random_seed=156,
        clip_grad_norm=1.1,
        validation=ValidateInTraining(
            data=Data(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=6,
            ),
            preprocessor=LocalizationPreprocessor(),
            interval=2
        )
    ),
    "dpn92_cls_train": Train(
        model=init_classifier(Resnet34ClassifierWrapper, Resnet34LocalizerWrapper, loc_version='0', loc_seed=0),
        version='1',
        seed=0,
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=12,
            num_workers=6,
        ),
        wrapper=Dpn92ClassifierWrapper(),
        preprocessor=Preprocessors.DPN92_CLS,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150,
                                                               170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.DPN92_CLS,
        score=Dpn92ClassifierWrapper.default_score,
        epochs=10,
        random_seed=54321,
        grad_scaling=True,
        clip_grad_norm=0.999,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6,
            ),
            preprocessor=ClassificationPreprocessor(),
            interval=2,
        )
    ),
    "dpn92_cls_tune": Train(
        model=lambda: Dpn92ClassifierWrapper().from_checkpoint(version='1', seed=0),
        version='tuned',
        seed=0,
        wrapper=Dpn92ClassifierWrapper(),
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=12,
            num_workers=6,
        ),
        preprocessor=Preprocessors.DPN92_CLS_TUNE,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.000008, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                                               90, 110, 130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.DPN92_CLS,
        score=Dpn92ClassifierWrapper.default_score,
        clip_grad_norm=0.999,
        random_seed=777,
        epochs=1,
        grad_scaling=True,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=4,
                num_workers=6,
            ),
            preprocessor=ClassificationPreprocessor(),
        )
    ),
    "senet154_loc_train": Train(
        model=lambda: SeNet154LocalizerWrapper().from_backbone(),
        version='1',
        seed=0,
        wrapper=SeNet154LocalizerWrapper(),
        data=Data(
            dataset=LocalizationDataset(train_dir),
            batch_size=14,
            num_workers=6,
        ),
        preprocessor=Preprocessors.SENET154_LOC,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[3, 7, 11, 15, 19, 23,
                                                               27, 33, 41, 50, 60,
                                                               70, 90, 110,
                                                               130, 150, 170, 180,
                                                               190],
                                                   gamma=0.5),
        loss=Losses.SENET154_LOC,
        score=SeNet154LocalizerWrapper.default_score,
        clip_grad_norm=0.999,
        epochs=30,
        random_seed=321,
        validation=ValidateInTraining(
            data=Data(
                dataset=LocalizationDataset(test_dir),
                batch_size=4,
                num_workers=6,
            ),
            preprocessor=LocalizationPreprocessor(),
            interval=1,
        )
    ),
    "senet154_cls_train": Train(
        model=init_classifier(SeNet154ClassifierWrapper, SeNet154LocalizerWrapper, loc_version='1', loc_seed='0'),
        version='1',
        seed=0,
        wrapper=SeNet154ClassifierWrapper(),
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=8,
            num_workers=6
        ),
        preprocessor=Preprocessors.DPN92_CLS,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[3, 5, 9, 13, 17, 21, 25, 29, 33, 47, 50, 60, 70, 90, 110,
                                                               130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.SENET154_CLS,
        score=SeResnext50ClassifierWrapper.default_score,
        clip_grad_norm=0.999,
        epochs=16,
        random_seed=123123,
        grad_scaling=True,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=2,
                num_workers=6
            ),
            preprocessor=ClassificationPreprocessor(),
            interval=2,
        )
    ),
    "senet154_cls_tune": Train(
        model=lambda: SeNet154ClassifierWrapper().from_checkpoint(version='1', seed=0),
        version='tuned',
        seed=0,
        wrapper=SeNet154ClassifierWrapper(),
        data=Data(
            dataset=ClassificationDataset(train_dir),
            batch_size=8,
            num_workers=6
        ),
        preprocessor=Preprocessors.DPN92_CLS_TUNE,
        optimizer=lambda model: AdamW(model.parameters(), lr=0.000008, weight_decay=1e-6),
        lr_scheduler=lambda optimizer: MultiStepLR(optimizer,
                                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70,
                                                               90, 110, 130, 150, 170, 180, 190],
                                                   gamma=0.5),
        loss=Losses.SENET154_CLS,
        score=SeNet154ClassifierWrapper.default_score,
        clip_grad_norm=0.999,
        epochs=2,
        random_seed=531,
        validation=ValidateInTraining(
            data=Data(
                dataset=ClassificationDataset(test_dir),
                batch_size=2,
                num_workers=6
            ),
            preprocessor=ClassificationPreprocessor(),
            interval=2,
        )
    )
}

if __name__ == '__main__':
    configs[sys.argv[1]].run()
