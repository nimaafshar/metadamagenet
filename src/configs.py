import pathlib
import enum


class DamageType:
    NO_DAMAGE = 1
    MINOR_DAMAGE = 2
    MAJOR_DAMAGE = 3
    DESTROYED = 4
    UN_CLASSIFIED = 1


damage_to_damage_type = {
    "no-damage": DamageType.NO_DAMAGE,
    "minor-damage": DamageType.MINOR_DAMAGE,
    "major-damage": DamageType.MAJOR_DAMAGE,
    "destroyed": DamageType.DESTROYED,
    "un-classified": DamageType.UN_CLASSIFIED
}

damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1
}

# DIRECTORY SETTINGS

TRAIN_DIRS = (
    pathlib.Path('./data/train'),
    pathlib.Path('./data/tier3')
)

TEST_DIR: pathlib.Path = pathlib.Path('./data/test/images')

TRAIN_SPLIT: pathlib.Path = pathlib.Path('./data/split/train/')
VALIDATION_SPLIT: pathlib.Path = pathlib.Path('./data/split/train/')

IMAGES_DIRECTORY = 'images'
LABELS_DIRECTORY = 'labels'
MASKS_DIRECTORY = 'masks'
LOCALIZATION_PREDICTION_MASKS_DIRECTORY = 'pred_loc_val'

PREDICTIONS_DIRECTORY = pathlib.Path('./data/pred/')

MODELS_WEIGHTS_FOLDER = pathlib.Path('./data/weights')

SUBMISSIONS_DIR = pathlib.Path('./data/submissions/')

# masks directory name
MASKS_DIR = 'masks'
