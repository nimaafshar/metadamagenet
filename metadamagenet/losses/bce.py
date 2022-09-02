import torch


class StableBCELoss(torch.nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.float().view(-1)
        targets = targets.float().view(-1)
        neg_abs = - outputs.abs()
        loss = outputs.clamp(min=0) - outputs * targets + (1 + neg_abs.exp()).log()
        return loss.mean()
