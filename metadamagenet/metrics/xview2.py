from .damage import DamageClassificationMetric, DamageLocalizationMetric
from torchmetrics import Dice

localization_score = Dice(multiclass=False, threshold=0.5, zero_division=0, average='micro')
classification_score = DamageClassificationMetric() * 0.7 + DamageLocalizationMetric() * 0.3
