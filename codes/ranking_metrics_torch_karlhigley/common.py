import torch


def _check_inputs(ks, scores, labels):
    if len(ks.shape) > 1:
        raise ValueError("ks should be a 1-dimensional tensor")

    if len(scores.shape) != 2:
        raise ValueError("scores must be a 2-dimensional tensor")

    if len(labels.shape) != 2:
        raise ValueError("labels must be a 2-dimensional tensor")

    if scores.shape != labels.shape:
        raise ValueError("scores and labels must be the same shape")

    return (
        ks.to(dtype=torch.int32, device=scores.device),
        scores.to(dtype=torch.float64, device=scores.device),
        labels.to(dtype=torch.float64, device=scores.device),
    )


def _extract_topk(ks, scores, labels):
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 1, topk_indices)
    return topk_scores, topk_indices, topk_labels


def _create_output_placeholder(scores, ks):
    return torch.zeros(scores.shape[0], len(ks)).to(
        device=scores.device, dtype=torch.float64
    )
