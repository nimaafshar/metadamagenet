from .f1 import LocalizationF1Score, DamageF1Score
from .weighted import WeightedImageMetric

classification_score = WeightedImageMetric(
    ("F1Loc", LocalizationF1Score(), 0.3),
    ("F1Damage", DamageF1Score(clip_localization_mask=True), 0.7)
)

localization_score = LocalizationF1Score()
