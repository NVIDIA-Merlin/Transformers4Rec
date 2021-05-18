#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict, List

import numpy as np
import torch

from .ranking_metrics_cpu import MRR as C_MRR
from .ranking_metrics_cpu import NDCG as C_NDCG
from .ranking_metrics_cpu import HitRate as C_HitRate
from .ranking_metrics_torch.avg_precision import avg_precision_at
from .ranking_metrics_torch.cumulative_gain import ndcg_at
from .ranking_metrics_torch.precision_recall import precision_at, recall_at

METRICS_MAPPING = {
    "ndcg": ndcg_at,
    "map": avg_precision_at,
    "recall": recall_at,
    "precision": precision_at,
}


class EvalMetrics(object):
    def __init__(self, ks=[5, 10, 100, 1000], use_cpu=False, use_torch=True):
        self.ks = ks
        self.max_k = max(ks)

        self.use_cpu = use_cpu
        self.use_torch = use_torch
        self.f_measures_cpu = []
        if use_cpu:
            f_ndcg_c = {f"ndcg@{k}": C_NDCG(k) for k in ks}
            f_recall_c = {f"recall@{k}": C_HitRate(k) for k in ks}
            f_mrr_c = {f"mrr@{k}": C_MRR(k) for k in ks}

            self.f_measures_cpu.extend([f_ndcg_c, f_recall_c, f_mrr_c])

        self.f_measures_torch = []
        if use_torch:
            f_precision_kh = MetricWrapper("precision", precision_at, ks)
            f_recall_kh = MetricWrapper("recall", recall_at, ks)
            f_avgp_kh = MetricWrapper("avg_precision", avg_precision_at, ks)
            f_ndcg_kh = MetricWrapper("ndcg", ndcg_at, ks)
            self.f_measures_torch.extend(
                [f_precision_kh, f_recall_kh, f_avgp_kh, f_ndcg_kh]
            )

    def reset(self):
        if self.use_torch:
            for f_measure in self.f_measures_torch:
                f_measure.reset()

        if self.use_cpu:
            for f_measure_ks in self.f_measures_cpu:
                for name, f_measure in f_measure_ks.items():
                    f_measure.reset()

    def update(self, preds, labels, return_individual_metrics=False):
        metrics_results = {}
        if self.use_torch:
            # compute metrics on PyTorch
            labels = torch.nn.functional.one_hot(
                labels.reshape(-1), preds.size(-1)
            ).detach()
            preds = preds.view(-1, preds.size(-1))
            for f_measure in self.f_measures_torch:
                results = f_measure.add(
                    preds, labels, return_individual_metrics=return_individual_metrics
                )
                # Merging metrics results
                if return_individual_metrics:
                    metrics_results = {**metrics_results, **results}

        if self.use_cpu:
            # compute metrics on CPU
            preds_cpu = preds
            labels_cpu = labels
            if type(preds_cpu) is torch.Tensor:
                preds_cpu = preds_cpu.cpu().numpy()
                labels_cpu = labels_cpu.cpu().numpy()

            # Gets only the top-k items (sorted by relevance) from the predictions for each label
            pred_items_sorted = np.argpartition(
                preds_cpu, kth=np.arange(-self.max_k, 0, 1), axis=-1
            )[:, -self.max_k :][:, ::-1]

            for f_measure_ks in self.f_measures_cpu:
                for name, f_measure in f_measure_ks.items():
                    f_measure.add(
                        np.expand_dims(pred_items_sorted, 0),
                        np.expand_dims(labels_cpu, 0),
                    )

        return metrics_results

    def result(self):
        metrics = []

        # CPU
        metrics.extend(
            [
                {name: f_measure.result() for name, f_measure in f_measure_ks.items()}
                for f_measure_ks in self.f_measures_cpu
            ]
        )

        # PyTorch
        metrics.extend([f_measure.result() for f_measure in self.f_measures_torch])

        return {k: v for d in metrics for k, v in d.items()}


class MetricWrapper(object):
    def __init__(self, name, f_metric, topks):
        self.name = name
        self.topks = topks
        self.f_metric = f_metric
        self.reset()

    def reset(self):
        self.results = {k: [] for k in self.topks}

    def add(self, predictions, labels, return_individual_metrics=False):

        # represent target class id as one-hot vector
        # labels = torch.nn.functional.one_hot(labels, predictions.size(-1)).detach()

        # Computing the metrics at different cut-offs
        metric = self.f_metric(torch.LongTensor(self.topks), predictions, labels)

        # del(labels)

        # Retrieving individual metric results (for each next-item recommendation list), to return for debug logging purposes
        if return_individual_metrics:
            returns = {}
            for k, measures in zip(self.topks, metric.T):
                returns[f"{self.name}@{k}"] = measures.cpu().numpy()

        # Computing the mean of the batch metrics (for each cut-off at topk)
        metric_mean = metric.mean(0)

        # Storing in memory the average metric results for this batch
        for k, measure in zip(self.topks, metric_mean):
            self.results[k].append(measure.cpu().item())

        if return_individual_metrics:
            # Returning individual metric results, for debug logging purposes
            return returns

    def result(self):
        return {f"{self.name}@{k}": np.mean(self.results[k]) for k in self.topks}


def compute_accuracy_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    metrics: str = ["ndcg", "map", "recall", "precision"],
    top_k: List = [5, 10, 100, 1000],
    average_metrics: bool = True,
) -> Dict:

    # flatten (n_batch , seq_len , n_events) to ((n_batch x seq_len), n_events)
    n_labels = preds.size(-1)
    preds = preds.view(-1, n_labels)
    labels = labels.reshape(-1)
    # represent target class id as one-hot vector
    labels = torch.nn.functional.one_hot(labels, n_labels)

    top_k = torch.LongTensor(top_k)
    metric_results = {}
    for metric_name in metrics:
        metric = METRICS_MAPPING[metric_name]
        # Computing the metrics at different cut-offs
        results_by_topk = metric(top_k, preds, labels)

        if average_metrics:
            results_by_topk = results_by_topk.mean(0)

        # Separating metrics for different top-k and converting to numpy
        for k, measures in zip(top_k, results_by_topk.T):
            metric_results[f"{metric_name}@{k}"] = measures.cpu().numpy()

    return metric_results
