import pathlib

damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1  # ?
}

TRAIN_DIRS = (
    pathlib.Path('./train'),
    pathlib.Path('./tier3')
)

# masks directory name
MASKS_DIR = 'masks'
