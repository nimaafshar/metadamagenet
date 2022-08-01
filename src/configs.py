import pathlib
import enum


class DamageType:
    NO_DAMAGE = 1
    MINOR_DAMAGE = 2
    MAJOR_DAMAGE = 3
    DESTROYED = 4
    UN_CLASSIFIED = 1


damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1  # ?
}

TRAIN_DIRS = (
    pathlib.Path('./data/train'),
    pathlib.Path('./data/tier3')
)

TEST_DIR = pathlib.Path('./test/images')

MODELS_FOLDER = pathlib.Path('./weights')

# masks directory name
MASKS_DIR = 'masks'
