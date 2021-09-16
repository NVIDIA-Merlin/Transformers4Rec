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

# Adapted from source code: https://github.com/karlhigley/ranking-metrics-torch
from abc import abstractmethod

import torch
import torchmetrics as tm

from merlin_standard_lib import Registry

from .utils import torch_utils

ranking_metrics_registry = Registry.class_registry("torch.ranking_metrics")


class RankingMetric(tm.Metric):
    """
    Metric wrapper for computing ranking metrics@K for session-based task.

    Parameters
    ----------
    top_ks : list, default [2, 5])
        list of cutoffs
    labels_onehot : bool
        Enable transform the labels to one-hot representation
    """

    def __init__(self, top_ks=None, labels_onehot=False):
        super(RankingMetric, self).__init__()
        self.top_ks = top_ks or [2, 5]
        self.labels_onehot = labels_onehot
        # Store the mean of the batch metrics (for each cut-off at topk)
        self.add_state("metric_mean", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):  # type: ignore
        # Computing the metrics at different cut-offs
        if self.labels_onehot:
            target = torch_utils.tranform_label_to_onehot(target, preds.size(-1))
        metric = self._metric(torch.LongTensor(self.top_ks), preds.view(-1, preds.size(-1)), target)
        self.metric_mean.append(metric)  # type: ignore

    def compute(self):
        # Computing the mean of the batch metrics (for each cut-off at topk)
        return torch.cat(self.metric_mean, axis=0).mean(0)

    @abstractmethod
    def _metric(self, ks: torch.Tensor, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute a ranking metric over a predictions and one-hot targets.
        This method should be overridden by subclasses.
        """


@ranking_metrics_registry.register_with_multiple_names("precision_at", "precision")
class PrecisionAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(PrecisionAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(self, ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute precision@K for each of the provided cutoffs

        Parameters
        ----------
        ks : torch.Tensor or list
            list of cutoffs
        scores : torch.Tensor
            predicted item scores
        labels : torch.Tensor
            true item labels

        Returns
        -------
        torch.Tensor:
            list of precisions at cutoffs
        """

        ks, scores, labels = torch_utils.check_inputs(ks, scores, labels)
        _, _, topk_labels = torch_utils.extract_topk(ks, scores, labels)
        precisions = torch_utils.create_output_placeholder(scores, ks)

        for index, k in enumerate(ks):
            precisions[:, index] = torch.sum(topk_labels[:, : int(k)], dim=1) / float(k)

        return precisions


@ranking_metrics_registry.register_with_multiple_names("recall_at", "recall")
class RecallAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(RecallAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(self, ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute recall@K for each of the provided cutoffs

        Parameters
        ----------
        ks : torch.Tensor or list
            list of cutoffs
        scores : torch.Tensor
            predicted item scores
        labels : torch.Tensor
            true item labels

        Returns
        -------
            torch.Tensor: list of recalls at cutoffs
        """

        ks, scores, labels = torch_utils.check_inputs(ks, scores, labels)
        _, _, topk_labels = torch_utils.extract_topk(ks, scores, labels)
        recalls = torch_utils.create_output_placeholder(scores, ks)

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


@ranking_metrics_registry.register_with_multiple_names("avg_precision_at", "avg_precision", "map")
class AvgPrecisionAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(AvgPrecisionAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)
        self.precision_at = PrecisionAt(top_ks)._metric

    def _metric(self, ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute average precision at K for provided cutoffs

        Parameters
        ----------
        ks : torch.Tensor or list
            list of cutoffs
        scores : torch.Tensor
            2-dim tensor of predicted item scores
        labels : torch.Tensor
            2-dim tensor of true item labels

        Returns
        -------
        torch.Tensor:
            list of average precisions at cutoffs
        """
        ks, scores, labels = torch_utils.check_inputs(ks, scores, labels)
        topk_scores, _, topk_labels = torch_utils.extract_topk(ks, scores, labels)
        avg_precisions = torch_utils.create_output_placeholder(scores, ks)

        # Compute average precisions at K
        num_relevant = torch.sum(labels, dim=1)
        max_k = ks.max().item()

        precisions = self.precision_at(1 + torch.arange(max_k), topk_scores, topk_labels)
        rel_precisions = precisions * topk_labels

        for index, k in enumerate(ks):
            total_prec = rel_precisions[:, : int(k)].sum(dim=1)
            avg_precisions[:, index] = total_prec / num_relevant.clamp(min=1, max=k).to(
                dtype=torch.float32, device=scores.device
            )  # Ensuring type is double, because it can be float if --fp16

        return avg_precisions


@ranking_metrics_registry.register_with_multiple_names("dcg_at", "dcg")
class DCGAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(DCGAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(
        self, ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2
    ) -> torch.Tensor:
        """Compute discounted cumulative gain at K for provided cutoffs (ignoring ties)

        Parameters
        ----------
        ks : torch.Tensor or list
            list of cutoffs
        scores : torch.Tensor
            predicted item scores
        labels : torch.Tensor
            true item labels

        Returns
        -------
        torch.Tensor :
            list of discounted cumulative gains at cutoffs
        """
        ks, scores, labels = torch_utils.check_inputs(ks, scores, labels)
        topk_scores, topk_indices, topk_labels = torch_utils.extract_topk(ks, scores, labels)
        dcgs = torch_utils.create_output_placeholder(scores, ks)

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


@ranking_metrics_registry.register_with_multiple_names("ndcg_at", "ndcg")
class NDCGAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(NDCGAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)
        self.dcg_at = DCGAt(top_ks)._metric

    def _metric(
        self, ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2
    ) -> torch.Tensor:
        """Compute normalized discounted cumulative gain at K for provided cutoffs (ignoring ties)

        Parameters
        ----------
        ks : torch.Tensor or list
            list of cutoffs
        scores : torch.Tensor
            predicted item scores
        labels : torch.Tensor
            true item labels

        Returns
        -------
        torch.Tensor :
            list of discounted cumulative gains at cutoffs
        """
        ks, scores, labels = torch_utils.check_inputs(ks, scores, labels)
        topk_scores, topk_indices, topk_labels = torch_utils.extract_topk(ks, scores, labels)
        # ndcgs = _create_output_placeholder(scores, ks) #TODO track if this line is needed

        # Compute discounted cumulative gains
        gains = self.dcg_at(ks, topk_scores, topk_labels)
        normalizing_gains = self.dcg_at(ks, topk_labels, topk_labels)

        # Prevent divisions by zero
        relevant_pos = (normalizing_gains != 0).nonzero(as_tuple=True)
        irrelevant_pos = (normalizing_gains == 0).nonzero(as_tuple=True)

        gains[irrelevant_pos] = 0
        gains[relevant_pos] /= normalizing_gains[relevant_pos]

        return gains
