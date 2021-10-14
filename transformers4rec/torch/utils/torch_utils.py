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

import abc
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from merlin_standard_lib import Schema
from merlin_standard_lib.utils.proto_utils import has_field

from ...config.schema import SchemaMixin
from ..typing import TabularData


class OutputSizeMixin(SchemaMixin, abc.ABC):
    def build(self, input_size, schema=None, **kwargs):
        self.check_schema(schema=schema)

        self.input_size = input_size
        if schema and not getattr(self, "schema", None):
            self.schema = schema

        return self

    def output_size(self, input_size=None):
        input_size = input_size or getattr(self, "input_size", None)
        if not input_size:
            # TODO: log warning here
            return None

        return self.forward_output_size(input_size)

    def forward_output_size(self, input_size):
        raise NotImplementedError()

    def __rrshift__(self, other):
        from ..block.base import right_shift_block

        return right_shift_block(self, other)


class LossMixin:
    """Mixin to use for `torch.Module`s that can calculate a loss."""

    def compute_loss(
        self,
        inputs: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
        compute_metrics: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the loss on a batch of data.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            TODO
        targets: Union[torch.Tensor, TabularData]
            TODO
        compute_metrics: bool, default=True
            Boolean indicating whether or not to update the state of the metrics
            (if they are defined).
        """
        raise NotImplementedError()


class MetricsMixin:
    """Mixin to use for `torch.Module`s that can calculate metrics."""

    def calculate_metrics(
        self,
        inputs: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
        mode: str = "val",
        forward=True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Calculate metrics on a batch of data, each metric is stateful and this updates the state.

        The state of each metric can be retrieved by calling the `compute_metrics` method.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            TODO
        targets: Union[torch.Tensor, TabularData]
            TODO
        forward: bool, default True

        mode: str, default="val"

        """
        raise NotImplementedError()

    def compute_metrics(self, mode: str = None) -> Dict[str, Union[float, torch.Tensor]]:
        """Returns the current state of each metric.

        The state is typically updated each batch by calling the `calculate_metrics` method.

        Parameters
        ----------
        mode: str, default="val"

        Returns
        -------
        Dict[str, Union[float, torch.Tensor]]
        """
        raise NotImplementedError()

    def reset_metrics(self):
        """Reset all metrics."""
        raise NotImplementedError()


def requires_schema(module):
    module.REQUIRES_SCHEMA = True

    return module


def check_gpu(module):
    try:
        return next(module.parameters()).is_cuda
    except StopIteration:
        return False


def get_output_sizes_from_schema(schema: Schema, batch_size=-1, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
        # Sequential or multi-hot feature
        if has_field(feature, "value_count"):
            sizes[name] = torch.Size(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        elif has_field(feature, "shape"):
            sizes[name] = torch.Size([batch_size] + [d.size for d in feature.shape.dim])
        else:
            sizes[name] = torch.Size([batch_size])

    return sizes


def calculate_batch_size_from_input_size(input_size):
    if isinstance(input_size, dict):
        input_size = [i for i in input_size.values() if isinstance(i, torch.Size)][0]

    return input_size[0]


def check_inputs(ks, scores, labels):
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
        scores,
        labels,  # .to(dtype=torch.float32, device=scores.device),
    )


def extract_topk(ks, scores, labels):
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 1, topk_indices)
    return topk_scores, topk_indices, topk_labels


def create_output_placeholder(scores, ks):
    return torch.zeros(scores.shape[0], len(ks)).to(device=scores.device, dtype=torch.float32)


def tranform_label_to_onehot(labels, vocab_size):
    return one_hot_1d(labels.reshape(-1), vocab_size, dtype=torch.float32).detach()


def one_hot_1d(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    r"""Coverts a 1d label tensor to one-hot representation

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: torch.float32

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([0, 1, 2, 0])
        >>> one_hot_1d(labels, num_classes=3)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.],
               ])
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    if not len(labels.shape) == 1:
        raise ValueError("Expected tensor should have 1 dim. Got: {}".format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype)
        )
    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one." " Got: {}".format(num_classes)
        )
    if device is None:
        device = labels.device
    labels_size = labels.shape[0]
    one_hot = torch.zeros(labels_size, num_classes, device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(-1), 1.0)


class LambdaModule(torch.nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        import types

        assert isinstance(lambda_fn, types.LambdaType)
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)


@dataclass
class MappingTransformerMasking:
    from transformers4rec.torch.masking import (
        CausalLanguageModeling,
        MaskedLanguageModeling,
        PermutationLanguageModeling,
        ReplacementLanguageModeling,
    )

    DEFAULT_MASKING = [
        CausalLanguageModeling,
        MaskedLanguageModeling,
        ReplacementLanguageModeling,
        PermutationLanguageModeling,
    ]

    BertConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    ConvBertConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    DebertaConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    DistilBertConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    GPT2Config = [CausalLanguageModeling]
    LongformerConfig = [CausalLanguageModeling, MaskedLanguageModeling, ReplacementLanguageModeling]
    MegatronBertConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    MPNetConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    RobertaConfig = [MaskedLanguageModeling, ReplacementLanguageModeling]
    RoFormerConfig = [CausalLanguageModeling, MaskedLanguageModeling, ReplacementLanguageModeling]
    TransfoXLConfig = [CausalLanguageModeling]
    XLNetConfig = [
        CausalLanguageModeling,
        MaskedLanguageModeling,
        ReplacementLanguageModeling,
        PermutationLanguageModeling,
    ]
