import torch

from .common import _check_inputs, _create_output_placeholder, _extract_topk
from .precision_recall import precision_at


def avg_precision_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute average precision at K for provided cutoffs

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): 2-dim tensor of predicted item scores
        labels (torch.Tensor): 2-dim tensor of true item labels

    Returns:
        torch.Tensor: list of average precisions at cutoffs
    """
    ks, scores, labels = _check_inputs(ks, scores, labels)
    topk_scores, _, topk_labels = _extract_topk(ks, scores, labels)
    avg_precisions = _create_output_placeholder(scores, ks)

    # Compute average precisions at K
    num_relevant = torch.sum(labels, dim=1)
    max_k = ks.max().item()

    precisions = precision_at(1 + torch.arange(max_k), topk_scores, topk_labels)
    rel_precisions = precisions * topk_labels

    for index, k in enumerate(ks):
        total_prec = rel_precisions[:, : int(k)].sum(dim=1)
        avg_precisions[:, index] = total_prec / num_relevant.clamp(min=1, max=k).to(
            dtype=torch.float32, device=scores.device
        )  # Ensuring type is double, because it can be float if --fp16

    return avg_precisions
