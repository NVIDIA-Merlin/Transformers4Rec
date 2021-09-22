import torch
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    """Constructor for cross-entropy loss with label smoothing

    Parameters:
    ----------
    smoothing: float
        The label smoothing factor. it should be between 0 and 1.
    weight: torch.Tensor
        The tensor of weights given to each class.
    reduction: str
        Specifies the reduction to apply to the output,
        possible values are `none` | `sum` | `mean`

    Adapted from https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch
    """

    def __init__(self, weight: torch.Tensor = None, reduction: str = "mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing: float = 0.0):
        assert 0 <= smoothing < 1, f"smoothing factor {smoothing} should be between 0 and 1"
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        lsm = inputs

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError(
                f"{self.reduction} is not supported, please choose one of the following values"
                " [`sum`, `none`, `mean`]"
            )
        return loss
