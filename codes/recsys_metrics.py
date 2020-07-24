
from typing import Dict, NamedTuple
import numpy as np
import torch

from chameleon_metrics import (
    NDCG as C_NDCG, 
    HitRate as C_HitRate, 
    StreamingMetric
)
from ranking_metrics_torch_karlhigley.precision_recall import precision_at, recall_at
from ranking_metrics_torch_karlhigley.avg_precision import avg_precision_at
from ranking_metrics_torch_karlhigley.cumulative_gain import ndcg_at
from recsys_utils import Timing


class EvalMetrics(object):
    def __init__(self, ks=[5, 10, 20, 50], use_cpu=False, use_gpu=True):
        
        f_ndcg_c = {f'ndcg_c@{k}': C_NDCG(k) for k in ks}
        f_recall_c = {f'recall_c@{k}': C_HitRate(k) for k in ks}

        f_precision_kh = MetricWrapper('precision', precision_at, ks)
        f_recall_kh = MetricWrapper('recall', recall_at, ks)
        f_avgp_kh = MetricWrapper('avg_precision', avg_precision_at, ks)
        f_ndcg_kh = MetricWrapper('ndcg', ndcg_at, ks)

        self.f_measures_cpu = []
        if use_cpu:
            self.f_measures_cpu.extend([
                f_ndcg_c, 
                f_recall_c
            ])

        self.f_measures_gpu = []
        if use_gpu:
            self.f_measures_gpu.extend([
                f_precision_kh,
                f_recall_kh,
                f_avgp_kh,
                f_ndcg_kh
            ])

    def update(self, preds, labels):

        # compute metrics on GPU
        for f_measure in self.f_measures_gpu:
            f_measure.add(*EvalMetrics.flatten(preds, labels))

        # compute metrics on CPU
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        for f_measure_ks in self.f_measures_cpu:
            for f_measure in f_measure_ks.values():
                f_measure.add(preds, labels)

    def result(self):
        metrics = [] 
        metrics.extend([{name: f_measure.result() \
            for name, f_measure in f_measure_ks.items()} \
                for f_measure_ks in self.f_measures_cpu]
        )
        metrics.extend([f_measure.result() for f_measure in self.f_measures_gpu])
        return {k: v for d in metrics for k, v in d.items()}

    @staticmethod
    def flatten(preds, labels):
        # flatten (n_batch x seq_len x n_events) to ((n_batch x seq_len) x n_events)
        preds = preds.view(-1, preds.size(-1))
        labels = labels.reshape(-1)
        
        return preds, labels


class MetricWrapper(object):
    def __init__(self, name, f_metric, topks):
        self.name = name
        self.topks = topks
        self.f_metric = f_metric
        self.reset()

    def reset(self):
        self.results = {k:[] for k in self.topks}

    def add(self, predictions, labels):

        # represent target class id as one-hot vector
        labels = torch.nn.functional.one_hot(labels, predictions.size(-1))

        metric = self.f_metric(torch.LongTensor(self.topks), predictions, labels)
        metric = metric.mean(0)
        for k, measure in zip(self.topks, metric):
            self.results[k].append(measure.cpu().item())

    def result(self):
        return {f'{self.name}@{k}': np.mean(self.results[k]) for k in self.topks}


