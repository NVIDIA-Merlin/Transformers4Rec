import torch


def LabelSmoothCrossEntropyLoss(smoothing: float = 0.0, reduction: str = "mean", **kwargs):
    """Coss-entropy loss with label smoothing.
    This is going to be deprecated. You should use torch.nn.CrossEntropyLoss()
    directly that in recent PyTorch versions already supports label_smoothing arg

    Parameters
    ----------
    smoothing: float
        The label smoothing factor. Specify a value between 0 and 1.
    reduction: str
        Specifies the reduction to apply to the output.
        Specify one of `none`, `sum`, or `mean`.

    Adapted from https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch
    """

    return torch.nn.CrossEntropyLoss(label_smoothing=smoothing, reduction=reduction, **kwargs)
