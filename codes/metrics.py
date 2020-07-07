
from typing import Dict, NamedTuple
import numpy as np
import torch

from ranking_metrics_torch_karlhigley.precision_recall import precision_at, recall_at
from ranking_metrics_torch_karlhigley.avg_precision import avg_precision_at
from ranking_metrics_torch_karlhigley.cumulative_gain import ndcg_at
from sklearn.metrics import ndcg_score, average_precision_score
from chameleon_metrics import NDCG, HitRate

from utils import Timing

# TODO: use this metrics
# https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/nar/metrics.py


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
    predictions = p.predictions.view(-1, event_size)
    labels = p.label_ids.reshape(-1)
    labels = torch.nn.functional.one_hot(labels, event_size).to(device)

    # metrics by Karl Higley's codde
    with Timing('gpu (Karl Higley) eval metrics computation'):
        rec_k = recall_at(_ks, predictions, labels)
        prec_k = precision_at(_ks, predictions, labels)
        avgp_k = avg_precision_at(_ks, predictions, labels)
        ndcg_k = ndcg_at(_ks, predictions, labels)

        rec_k = {"recall_{}".format(k): measure.mean() for k, measure in zip(ks, rec_k)}
        prec_k = {"precision_{}".format(k): measure.mean() for k, measure in zip(ks, prec_k)}
        avgp_k = {"avgprec_{}".format(k): measure.mean() for k, measure in zip(ks, avgp_k)}
        ndcg_k = {"ndcg_{}".format(k): measure.mean() for k, measure in zip(ks, ndcg_k)}

    labels_cpu, predictions_cpu = labels.cpu().numpy(), predictions.cpu().numpy()

    # metrics by Scikit-learn
    with Timing('cpu (scikit-learn) eval metrics computation'):
        ndcg_sci_k ={"ndcg_s_{}".format(k): ndcg_score(labels_cpu, predictions_cpu, k=k) for k in ks}
        ap_score_sci = {"avgp_s": average_precision_score(labels_cpu, predictions_cpu)}
    return {**rec_k, **prec_k, **avgp_k, **ndcg_k, **ndcg_sci_k, **ap_score_sci}

