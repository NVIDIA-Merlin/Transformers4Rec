import torch

from .common import _check_inputs, _create_output_placeholder, _extract_topk


def dcg_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2
) -> torch.Tensor:
    """Compute discounted cumulative gain at K for provided cutoffs (ignoring ties)

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of discounted cumulative gains at cutoffs
    """
    ks, scores, labels = _check_inputs(ks, scores, labels)
    topk_scores, topk_indices, topk_labels = _extract_topk(ks, scores, labels)
    dcgs = _create_output_placeholder(scores, ks)

    # Compute discounts
    discount_positions = torch.arange(ks.max().item()).to(
        device=scores.device, dtype=torch.float32
    )

    discount_log_base = torch.log(
        torch.Tensor([log_base]).to(device=scores.device, dtype=torch.float32)
    ).item()

    discounts = 1 / (torch.log(discount_positions + 2) / discount_log_base)

    # Compute DCGs at K
    for index, k in enumerate(ks):
        dcgs[:, index] = torch.sum(
            (topk_labels[:, :k] * discounts[:k].repeat(topk_labels.shape[0], 1)), dim=1
        ).to(
            dtype=torch.float32, device=scores.device
        )  # Ensuring type is double, because it can be float if --fp16

    return dcgs


def ndcg_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2
) -> torch.Tensor:
    """Compute normalized discounted cumulative gain at K for provided cutoffs (ignoring ties)

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of discounted cumulative gains at cutoffs
    """
    ks, scores, labels = _check_inputs(ks, scores, labels)
    topk_scores, topk_indices, topk_labels = _extract_topk(ks, scores, labels)
    ndcgs = _create_output_placeholder(scores, ks)

    # Compute discounted cumulative gains
    gains = dcg_at(ks, topk_scores, topk_labels)
    normalizing_gains = dcg_at(ks, topk_labels, topk_labels)

    # Prevent divisions by zero
    relevant_pos = (normalizing_gains != 0).nonzero(as_tuple=True)
    irrelevant_pos = (normalizing_gains == 0).nonzero(as_tuple=True)

    gains[irrelevant_pos] = 0
    gains[relevant_pos] /= normalizing_gains[relevant_pos]

    return gains
