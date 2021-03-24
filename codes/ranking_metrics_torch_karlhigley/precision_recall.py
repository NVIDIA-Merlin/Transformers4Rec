import torch

from .common import _check_inputs, _create_output_placeholder, _extract_topk


def precision_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute precision@K for each of the provided cutoffs

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of precisions at cutoffs
    """

    ks, scores, labels = _check_inputs(ks, scores, labels)
    _, _, topk_labels = _extract_topk(ks, scores, labels)
    precisions = _create_output_placeholder(scores, ks)

    for index, k in enumerate(ks):
        precisions[:, index] = torch.sum(topk_labels[:, : int(k)], dim=1) / float(k)

    return precisions


def recall_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute recall@K for each of the provided cutoffs

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of recalls at cutoffs
    """

    ks, scores, labels = _check_inputs(ks, scores, labels)
    _, _, topk_labels = _extract_topk(ks, scores, labels)
    recalls = _create_output_placeholder(scores, ks)

    # Compute recalls at K
    num_relevant = torch.sum(labels, dim=-1)
    rel_indices = (num_relevant != 0).nonzero()
    rel_count = num_relevant[rel_indices].squeeze()

    if rel_indices.shape[0] > 0:
        for index, k in enumerate(ks):
            rel_labels = topk_labels[rel_indices, : int(k)].squeeze()

            recalls[rel_indices, index] = (
                torch.div(torch.sum(rel_labels, dim=-1), rel_count)
                .reshape(len(rel_indices), 1)
                .to(dtype=torch.float32)
            )  # Ensuring type is double, because it can be float if --fp16

    return recalls


def _test_at():
    scores = torch.arange(0, 1, step=0.1).expand((1, 10))
    ks = torch.LongTensor([1, 2, 3, 10])
    print("scores:{}".format(scores))
    print("ks:{}".format(ks))

    print("-" * 10 + "\ntest1")
    labels = torch.LongTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]).expand((1, 10))
    print("label: {}".format(labels))

    result = recall_at(ks, scores, labels)
    print("recall:{}".format(result))

    result = precision_at(ks, scores, labels)
    print("precision:{}".format(result))

    print("-" * 10 + "\ntest2")

    labels = torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).expand((1, 10))
    print("label: {}".format(labels))

    result = recall_at(ks, scores, labels)
    print("recall:{}".format(result))

    result = precision_at(ks, scores, labels)
    print("precision:{}".format(result))


if __name__ == "__main__":
    _test_at()
