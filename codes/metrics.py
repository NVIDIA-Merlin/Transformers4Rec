
from typing import Dict, NamedTuple
import numpy as np
import torch

from ranking_metrics_torch_karlhigley.precision_recall import precision_at, recall_at
from ranking_metrics_torch_karlhigley.avg_precision import avg_precision_at
from ranking_metrics_torch_karlhigley.cumulative_gain import ndcg_at



class EvalPredictionTensor(NamedTuple):
    """
    Evaluation output (always contains labels), to be used
    to compute metrics.
    """

    predictions: torch.Tensor
    label_ids: torch.Tensor


def compute_recsys_metrics(p: EvalPredictionTensor, ks=[5, 10, 20, 50, 100, 10000]) -> Dict:
    
    # NOTE: currently under construction 

    device = p.predictions.device
    event_size = p.predictions.size(-1)

    _ks = torch.LongTensor(ks).to(device)
    preds_mb = p.predictions.view(-1, event_size)
    labels = p.label_ids.reshape(-1)
    labels_onehot = torch.nn.functional.one_hot(labels, event_size).to(device)

    rec_k = recall_at(_ks, preds_mb, labels_onehot)
    prec_k = precision_at(_ks, preds_mb, labels_onehot)
    avgp_k = avg_precision_at(_ks, preds_mb, labels_onehot)
    ndcg_k = ndcg_at(_ks, preds_mb, labels_onehot)

    rec_k = {"recall_{}".format(k): measure.mean() for k, measure in zip(ks, rec_k)}
    prec_k = {"precision_{}".format(k): measure.mean() for k, measure in zip(ks, prec_k)}
    avgp_k = {"avgprec_{}".format(k): measure.mean() for k, measure in zip(ks, avgp_k)}
    ndcg_k = {"ndcg_{}".format(k): measure.mean() for k, measure in zip(ks, ndcg_k)}

    return {**rec_k, **prec_k, **avgp_k, **ndcg_k}

