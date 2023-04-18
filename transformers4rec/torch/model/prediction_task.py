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


import logging
from math import sqrt
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torchmetrics as tm

from ..block.base import Block, BuildableBlock, SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import MaskedLanguageModeling
from ..ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt
from ..utils.torch_utils import LambdaModule
from .base import BlockType, PredictionTask

LOG = logging.getLogger("transformers4rec")


class BinaryClassificationPrepareBlock(BuildableBlock):
    def build(self, input_size) -> SequentialBlock:
        return SequentialBlock(
            torch.nn.Linear(input_size[-1], 1, bias=False),
            torch.nn.Sigmoid(),
            LambdaModule(lambda x: torch.squeeze(x, -1)),
            output_size=[
                None,
            ],
        )


class BinaryClassificationTask(PredictionTask):
    """Returns a ``PredictionTask`` for binary classification.

    Example usage::

        # Define the input module to process the tabular input features.
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=d_model,
            aggregation="concat",
            masking=None,
        )

        # Define XLNetConfig class and set default parameters for HF XLNet config.
        transformer_config = tr.XLNetConfig.build(
            d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
        )

        # Define the model block including: inputs, masking, projection and transformer block.
        body = tr.SequentialBlock(
            input_module,
            tr.MLPBlock([64]),
            tr.TransformerBlock(
                transformer_config,
                masking=input_module.masking
            )
        )

        # Define a head with BinaryClassificationTask.
        head = tr.Head(
            body,
            tr.BinaryClassificationTask(
                "click",
                summary_type="mean",
                metrics=[
                    tm.Precision(task='binary'),
                    tm.Recall(task='binary'),
                    tm.Accuracy(task='binary'),
                    tm.F1Score(task='binary')
                ]
            ),
            inputs=input_module,
        )

        # Get the end-to-end Model class.
        model = tr.Model(head)

    Parameters
    ----------

    target_name: Optional[str] = None
        Specifies the variable name that represents the positive and negative values.

    task_name: Optional[str] = None
        Specifies the name of the prediction task. If this parameter is not specified,
        a name is automatically constructed based on ``target_name`` and the Python
        class name of the model.

    task_block: Optional[BlockType] = None
        Specifies a module to transform the input tensor before computing predictions.

    loss: torch.nn.Module
        Specifies the loss function for the task.
        The default class is ``torch.nn.BCELoss``.

    metrics: Tuple[torch.nn.Module, ...]
        Specifies the metrics to calculate during training and evaluation.
        The default metrics are ``Precision``, ``Recall``, and ``Accuracy``.

    summary_type: str
        Summarizes a sequence into a single tensor. Accepted values are:

            - ``last`` -- Take the last token hidden state (like XLNet)
            - ``first`` -- Take the first token hidden state (like Bert)
            - ``mean`` -- Take the mean of all tokens hidden states
            - ``cls_index`` -- Supply a Tensor of classification token position (GPT/GPT-2)
            - ``attn`` -- Not implemented now, use multi-head attention
    """

    DEFAULT_LOSS = torch.nn.BCELoss()
    DEFAULT_METRICS = (
        tm.Precision(num_classes=2, task="binary"),
        tm.Recall(num_classes=2, task="binary"),
        tm.Accuracy(task="binary"),
        # TODO: Fix this: tm.AUC()
    )

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[BlockType] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        self.target_dim = 1
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            pre=BinaryClassificationPrepareBlock(),
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )


class RegressionPrepareBlock(BuildableBlock):
    def build(self, input_size) -> SequentialBlock:
        return SequentialBlock(
            torch.nn.Linear(input_size[-1], 1),
            LambdaModule(lambda x: torch.squeeze(x, -1)),
            output_size=[
                None,
            ],
        )


class RegressionTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.MSELoss()
    DEFAULT_METRICS = (tm.regression.MeanSquaredError(),)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[BlockType] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        self.target_dim = 1
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            pre=RegressionPrepareBlock(),
        )


class NextItemPredictionTask(PredictionTask):
    """This block performs item prediction task for session and sequential-based models.
    It requires a body containing a masking schema to use for training and target generation.
    For the supported masking schemes, please refers to:
    https://nvidia-merlin.github.io/Transformers4Rec/main/model_definition.html#sequence-masking

    Parameters
    ----------
    loss: torch.nn.Module
        Loss function to use. Defaults to NLLLos.
    metrics: Iterable[torchmetrics.Metric]
        List of ranking metrics to use for evaluation.
    task_block:
        Module to transform input tensor before computing predictions.
    task_name: str, optional
        Name of the prediction task, if not provided a name will be automatically constructed based
        on the target-name & class-name.
    weight_tying: bool
        The item id embedding table weights are shared with the prediction network layer.
    softmax_temperature: float
        Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
        Value 1.0 reduces to regular softmax.
    padding_idx: int
        pad token id.
    target_dim: int
        vocabulary size of item ids
    sampled_softmax: Optional[bool]
        Enables sampled softmax. By default False
    max_n_samples: Optional[int]
        Number of samples for sampled softmax. By default 100
    """

    DEFAULT_METRICS = (
        # default metrics suppose labels are int encoded
        NDCGAt(top_ks=[10, 20], labels_onehot=True),
        AvgPrecisionAt(top_ks=[10, 20], labels_onehot=True),
        RecallAt(top_ks=[10, 20], labels_onehot=True),
    )

    def __init__(
        self,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        metrics: Iterable[tm.Metric] = DEFAULT_METRICS,
        task_block: Optional[BlockType] = None,
        task_name: str = "next-item",
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        padding_idx: int = 0,
        target_dim: int = None,
        sampled_softmax: Optional[bool] = False,
        max_n_samples: Optional[int] = 100,
    ):
        super().__init__(loss=loss, metrics=metrics, task_block=task_block, task_name=task_name)
        self.softmax_temperature = softmax_temperature
        self.weight_tying = weight_tying
        self.padding_idx = padding_idx
        self.target_dim = target_dim
        self.sampled_softmax = sampled_softmax
        self.max_n_samples = max_n_samples

        self.item_embedding_table = None
        self.masking = None

    def build(self, body, input_size, device=None, inputs=None, task_block=None, pre=None):
        """Build method, this is called by the `Head`."""
        if not len(input_size) == 3 or isinstance(input_size, dict):
            raise ValueError(
                "NextItemPredictionTask needs a 3-dim vector as input, found:" f"{input_size}"
            )

        # Retrieve the embedding module to get the name of itemid col and its related table
        if not inputs:
            inputs = body.inputs
        if not getattr(inputs, "item_id", None):
            raise ValueError(
                "For Item Prediction task a categorical_module "
                "including an item_id column is required."
            )
        self.embeddings = inputs.categorical_module
        if not self.target_dim:
            self.target_dim = self.embeddings.item_embedding_table.num_embeddings
        if self.weight_tying:
            self.item_embedding_table = self.embeddings.item_embedding_table
            item_dim = self.item_embedding_table.weight.shape[1]
            if input_size[-1] != item_dim and not task_block:
                LOG.warning(
                    f"Projecting inputs of NextItemPredictionTask to'{item_dim}' "
                    f"As weight tying requires the input dimension '{input_size[-1]}' "
                    f"to be equal to the item-id embedding dimension '{item_dim}'"
                )
                # project input tensors to same dimension as item-id embeddings
                task_block = MLPBlock([item_dim])

        # Retrieve the masking from the input block
        self.masking = inputs.masking
        if not self.masking:
            raise ValueError(
                "The input block should contain a masking schema for training and evaluation"
            )
        self.padding_idx = self.masking.padding_idx
        pre = NextItemPredictionPrepareBlock(
            target_dim=self.target_dim,
            weight_tying=self.weight_tying,
            item_embedding_table=self.item_embedding_table,
            softmax_temperature=self.softmax_temperature,
            sampled_softmax=self.sampled_softmax,
            max_n_samples=self.max_n_samples,
            min_id=self.padding_idx + 1,
        )
        super().build(
            body, input_size, device=device, inputs=inputs, task_block=task_block, pre=pre
        )

    def forward(self, inputs: torch.Tensor, targets=None, training=False, testing=False, **kwargs):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        x = inputs.float()

        if self.task_block:
            x = self.task_block(x)  # type: ignore

        # Retrieve labels from masking
        if training or testing:
            labels = self.masking.masked_targets  # type: ignore
            trg_flat = labels.flatten()
            non_pad_mask = trg_flat != self.padding_idx
            labels_all = torch.masked_select(trg_flat, non_pad_mask).long()
            # remove padded items, keep only masked positions
            x = self.remove_pad_3d(x, non_pad_mask)
            y = labels_all
            x, y = self.pre(x, targets=y, training=training, testing=testing)  # type: ignore

            loss = self.loss(x, y)
            return {
                "loss": loss,
                "labels": y,
                "predictions": x,
            }
        else:
            # Get the hidden position to use for predicting the next item
            labels = self.embeddings.item_seq
            non_pad_mask = labels != self.padding_idx
            rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
            if isinstance(self.masking, MaskedLanguageModeling):
                last_item_sessions = non_pad_mask.sum(dim=1)
            else:
                last_item_sessions = non_pad_mask.sum(dim=1) - 1
            x = x[rows_ids, last_item_sessions]

            # Compute predictions probs
            x, _ = self.pre(x)  # type: ignore

            return x

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor

    def calculate_metrics(self, predictions, targets) -> Dict[str, torch.Tensor]:  # type: ignore
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        outputs = {}
        predictions = self.forward_to_prediction_fn(predictions)

        for metric in self.metrics:
            outputs[self.metric_name(metric)] = metric(predictions, targets)

        return outputs

    def compute_metrics(self):
        metrics = {
            self.metric_name(metric): metric.compute()
            for metric in self.metrics
            if getattr(metric, "top_ks", None)
        }
        # Explode metrics for each cut-off
        # TODO make result generic:
        # To accept a mix of ranking metrics and others not requiring top_ks ?
        topks = {self.metric_name(metric): metric.top_ks for metric in self.metrics}
        results = {}
        for name, metric in metrics.items():
            for measure, k in zip(metric, topks[name]):
                results[f"{name}_{k}"] = measure
        return results


class NextItemPredictionPrepareBlock(BuildableBlock):
    def __init__(
        self,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
        sampled_softmax: Optional[bool] = False,
        max_n_samples: Optional[int] = 100,
        min_id: Optional[int] = 0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.sampled_softmax = sampled_softmax
        self.max_n_samples = max_n_samples
        self.min_id = min_id

    def build(self, input_size) -> Block:
        return Block(
            _NextItemPredictionTask(
                input_size,
                self.target_dim,
                self.weight_tying,
                self.item_embedding_table,
                self.softmax_temperature,
                self.sampled_softmax,
                self.max_n_samples,
                self.min_id,
            ),
            [-1, self.target_dim],
        )


class _NextItemPredictionTask(torch.nn.Module):
    """Predict the interacted item-id probabilities.

    - During inference, the task consists of predicting the next item.
    - During training, the class supports the following Language modeling tasks:
        Causal LM, Masked LM, Permutation LM and Replacement Token Detection

    Parameters:
    -----------
        input_size: int
            Input size of this module.
        target_dim: int
            Dimension of the target.
        weight_tying: bool
            The item id embedding table weights are shared with the prediction network layer.
        item_embedding_table: torch.nn.Module
            Module that's used to store the embedding table for the item.
        softmax_temperature: float
            Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
            Value 1.0 reduces to regular softmax.
        sampled_softmax: Optional[bool]
            Enables sampled softmax. By default False
        max_n_samples: Optional[int]
            Number of samples for sampled softmax. By default 100
        min_id : Optional[int]
            The minimum value of the range for the log-uniform sampling. By default 0.
    """

    def __init__(
        self,
        input_size: Sequence,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
        sampled_softmax: Optional[bool] = False,
        max_n_samples: Optional[int] = 100,
        min_id: Optional[int] = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.sampled_softmax = sampled_softmax

        if not self.weight_tying:
            self.output_layer = torch.nn.Parameter(torch.empty(self.target_dim, input_size[-1]))
            torch.nn.init.kaiming_uniform_(self.output_layer, a=sqrt(5))

        if self.sampled_softmax:
            self.sampler = LogUniformSampler(
                max_n_samples=max_n_samples,
                max_id=target_dim,
                min_id=min_id,
                unique_sampling=True,
            )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        training=False,
        testing=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.weight_tying:
            output_weights = self.item_embedding_table.weight
        else:
            output_weights = self.output_layer

        if self.sampled_softmax and training:
            logits, targets = self.sampled(inputs, targets, output_weights)
        else:
            logits = inputs @ output_weights.t()

        if self.softmax_temperature:
            # Softmax temperature to reduce model overconfidence
            # and better calibrate probs and accuracy
            logits = torch.div(logits, self.softmax_temperature)

        return logits, targets

    def sampled(self, inputs, targets, output_weights):
        """Returns logits using sampled softmax"""
        neg_samples, targets_probs, samples_probs = self.sampler.sample(targets)

        positive_weights = output_weights[targets]
        negative_weights = output_weights[neg_samples]

        positive_scores = (inputs * positive_weights).sum(dim=-1, keepdim=True)
        negative_scores = inputs @ negative_weights.t()

        # logQ correction, to not overpenalize popular items for being sampled
        # more often as negatives
        epsilon = 1e-16
        positive_scores -= torch.unsqueeze(torch.log(targets_probs + epsilon), dim=-1)
        negative_scores -= torch.unsqueeze(torch.log(samples_probs + epsilon), dim=0)

        # Remove accidental matches
        accidental_hits = torch.unsqueeze(targets, -1) == torch.unsqueeze(neg_samples, 0)
        negative_scores[accidental_hits] = torch.finfo(torch.float16).min / 100.0

        logits = torch.cat([positive_scores, negative_scores], axis=1)
        new_targets = torch.zeros(logits.shape[0], dtype=torch.int64, device=targets.device)

        return logits, new_targets

    def _get_name(self) -> str:
        return "NextItemPredictionTask"


class LogUniformSampler(torch.nn.Module):
    def __init__(
        self,
        max_n_samples: int,
        max_id: int,
        min_id: Optional[int] = 0,
        unique_sampling: bool = True,
        n_samples_multiplier_before_unique: int = 2,
    ):
        """LogUniformSampler samples negative samples based on a log-uniform distribution.
        `P(class) = (log(class + 2) - log(class + 1)) / log(max_id + 1)`

        This implementation is based on to:
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/log_uniform_sampler.py
        TensorFlow Reference:
        https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py

        LogUniformSampler assumes item ids are sorted decreasingly by their frequency.

        if `unique_sampling==True`, then only unique sampled items will be returned.
        The actual # samples will vary from run to run if `unique_sampling==True`,
        as sampling without replacement (`torch.multinomial(..., replacement=False)`) is slow,
        so we use `torch.multinomial(..., replacement=True).unique()` which doesn't guarantee
        the same number of unique sampled items. You can try to increase
        n_samples_multiplier_before_unique to increase the chances to have more
        unique samples in that case.

        Parameters
        ----------
        max_n_samples : int
            The maximum desired number of negative samples. The number of samples might be
            smaller than that if `unique_sampling==True`, as explained above.
        max_id : int
            The maximum value of the range for the log-uniform distribution.
        min_id : Optional[int]
            The minimum value of the range for the log-uniform sampling. By default 0.
        unique_sampling : bool
            Whether to return unique samples. By default True
        n_samples_multiplier_before_unique : int
            If unique_sampling=True, it is not guaranteed that the number of returned
            samples will be equal to max_n_samples, as explained above.
            You can increase n_samples_multiplier_before_unique to maximize
            chances that a larger number of unique samples is returned.
        """
        super().__init__()

        if max_id <= 0:
            raise ValueError("max_id must be a positive integer.")
        if max_n_samples <= 0:
            raise ValueError("n_sample must be a positive integer.")

        self.max_id = max_id
        self.unique_sampling = unique_sampling
        self.max_n_samples = max_n_samples
        self.n_sample = max_n_samples
        if self.unique_sampling:
            self.n_sample = int(self.n_sample * n_samples_multiplier_before_unique)

        with torch.no_grad():
            dist = self.get_log_uniform_distr(max_id, min_id)
            self.register_buffer("dist", dist)
            unique_sampling_dist = self.get_unique_sampling_distr(dist, self.n_sample)
            self.register_buffer("unique_sampling_dist", unique_sampling_dist)

    def get_log_uniform_distr(self, max_id: int, min_id: int = 0) -> torch.Tensor:
        """Approximates the items frequency distribution with log-uniform probability distribution
        with P(class) = (log(class + 2) - log(class + 1)) / log(max_id + 1).
        It assumes item ids are sorted decreasingly by their frequency.

        Parameters
        ----------
        max_id : int
            Maximum discrete value for sampling (e.g. cardinality of the item id)

        Returns
        -------
        torch.Tensor
            Returns the log uniform probability distribution
        """
        log_indices = torch.arange(1.0, max_id - min_id + 2.0, 1.0).log_()
        probs = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
        if min_id > 0:
            probs = torch.cat([torch.zeros([min_id], dtype=probs.dtype), probs], axis=0)
        return probs

    def get_unique_sampling_distr(self, dist, n_sample):
        """Returns the probability that each item is sampled at least once
        given the specified number of trials. This is meant to be used when
        self.unique_sampling == True.
        That probability can be approximated by by 1 - (1 - p)^n
        and we use a numerically stable version: -expm1(num_tries * log1p(-p))
        """
        return (-(-dist.double().log1p_() * n_sample).expm1_()).float()

    def sample(self, labels: torch.Tensor):
        """Sample negative samples and calculate their probabilities.

        If `unique_sampling==True`, then only unique sampled items will be returned.
        The actual # samples will vary from run to run if `unique_sampling==True`,
        as sampling without replacement (`torch.multinomial(..., replacement=False)`) is slow,
        so we use `torch.multinomial(..., replacement=True).unique()`
        which doesn't guarantee the same number of unique sampled items.
        You can try to increase n_samples_multiplier_before_unique
        to increase the chances to have more unique samples in that case.

        Parameters
        ----------
        labels : torch.Tensor, dtype=torch.long, shape=(batch_size,)
            The input labels for which negative samples should be generated.

        Returns
        -------
        neg_samples : torch.Tensor, dtype=torch.long, shape=(n_samples,)
            The unique negative samples drawn from the log-uniform distribution.
        true_probs : torch.Tensor, dtype=torch.float32, shape=(batch_size,)
            The probabilities of the input labels according
            to the log-uniform distribution (depends on self.unique_sampling choice).
        samp_log_probs : torch.Tensor, dtype=torch.float32, shape=(n_samples,)
            The probabilities of the sampled negatives according
            to the log-uniform distribution (depends on self.unique_sampling choice).
        """

        if not torch.is_tensor(labels):
            raise TypeError("Labels must be a torch.Tensor.")
        if labels.dtype != torch.long:
            raise ValueError("Labels must be a tensor of dtype long.")
        if labels.dim() > 2 or (labels.dim() == 2 and min(labels.shape) > 1):
            raise ValueError(
                "Labels must be a 1-dimensional tensor or a 2-dimensional tensor"
                "with one of the dimensions equal to 1."
            )
        if labels.size(0) == 0:
            raise ValueError("Labels must not be an empty tensor.")
        if (labels < 0).any() or (labels > self.max_id).any():
            raise ValueError("All label values must be within the range [0, max_id].")

        n_tries = self.n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()[
                : self.max_n_samples
            ]

            device = labels.device
            neg_samples = neg_samples.to(device)

            if self.unique_sampling:
                dist = self.unique_sampling_dist
            else:
                dist = self.dist

            true_probs = dist[labels]
            samples_probs = dist[neg_samples]

            return neg_samples, true_probs, samples_probs

    def forward(self, labels):
        return self.sample(labels)
