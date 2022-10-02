from torchmetrics import Dice

localization_score = Dice(multiclass=False, threshold=0.5, zero_division=0, average='micro')
