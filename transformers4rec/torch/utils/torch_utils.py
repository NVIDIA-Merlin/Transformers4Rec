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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
from merlin.schema import Schema as CoreSchema
from merlin.schema.io.proto_utils import has_field

from merlin_standard_lib import Schema
from merlin_standard_lib.schema.schema import ColumnSchema

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
    """Mixin to use for a `torch.Module` that can calculate a loss."""

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
    """Mixin to use for a `torch.Module` that can calculate metrics."""

    def calculate_metrics(
        self,
        inputs: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
    ) -> Dict[str, torch.Tensor]:
        """Calculate metrics on a batch of data, each metric is stateful and this updates the state.

        The state of each metric can be retrieved by calling the `compute_metrics` method.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            Tensor or dictionary of predictions returned by the T4Rec model
        targets: Union[torch.Tensor, TabularData]
            Tensor or dictionary of true labels returned by the T4Rec model


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


def _has_field(col_schema, field_name):
    if isinstance(col_schema, ColumnSchema):
        return has_field(col_schema, field_name)

    return getattr(col_schema, field_name, None)


def _get_size_from_shape(col_schema, batch_size) -> torch.Size:
    shape = [batch_size]

    if isinstance(col_schema, ColumnSchema):
        if has_field(col_schema, "shape"):
            shape += [d.size for d in col_schema.shape.dim]
    elif col_schema.shape.dims is not None:
        if len(col_schema.shape.dims) == 1 and col_schema.shape.dims[0].max is None:
            return torch.Size(shape)
        raise NotImplementedError("TODO: support shape.dims")

    return torch.Size(shape)


def get_output_sizes_from_schema(schema: Schema, batch_size=-1, max_sequence_length=None):
    sizes = {}

    features = schema if isinstance(schema, CoreSchema) else schema.feature

    for feature in features:
        name = feature.name
        # Sequential or multi-hot feature
        if _has_field(feature, "value_count"):
            sizes[name] = torch.Size(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        else:
            sizes[name] = _get_size_from_shape(feature, batch_size)

    return sizes


def calculate_batch_size_from_input_size(input_size):
    if isinstance(input_size, dict):
        input_size = [i for i in input_size.values() if isinstance(i, torch.Size)][0]

    return input_size[0]


def check_inputs(ks, scores, labels):
    if not (isinstance(ks, (list, tuple)) and len(ks) >= 1):
        raise ValueError("ks should be a list or tuple with at least one element")

    if len(scores.shape) != 2:
        raise ValueError("scores must be a 2-dimensional tensor")

    if len(labels.shape) != 2:
        raise ValueError("labels must be a 2-dimensional tensor")

    if scores.shape != labels.shape:
        raise ValueError("scores and labels must be the same shape")

    return (
        ks,
        scores,
        labels,
    )


def extract_topk(ks, scores, labels):
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 1, topk_indices)
    return topk_scores, topk_indices, topk_labels


def create_output_placeholder(scores, ks):
    return torch.zeros(scores.shape[0], len(ks)).to(device=scores.device, dtype=torch.float32)


def tranform_label_to_onehot(labels, vocab_size):
    return one_hot_1d(labels.reshape(-1).to(torch.int64), vocab_size, dtype=torch.float32).detach()


def nested_detach(tensors):
    """Detach `tensors` (even if it's a nested list/tuple/dict of tensors).
    #TODO this method was copied from the latest version of HF transformers library to support
    dict outputs. So we should remove it when T4Rec is updated to use the latest version
    """
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed.
    Works for tensors or nested list/tuples/dict of tensors.
    #TODO this method was copied from the latest version of HF transformers library to support
    dict outputs. So we should remove it when T4Rec is updated to use the latest version
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)}"
    f" and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(
            nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors)
        )
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {
                k: nested_concat(t, new_tensors[k], padding_index=padding_index)
                for k, t in tensors.items()
            }
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second as needed

    #TODO this method was copied from the latest version of HF transformers library to support
    dict outputs. So we should remove it when T4Rec is updated to use the latest version
    """
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (
        tensor1.shape[0] + tensor2.shape[0],
        max(tensor1.shape[1], tensor2.shape[1]),
    ) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result


def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def nested_numpify(tensors):
    """Numpify `tensors` (even if it's a nested list/tuple/dict of tensors).
    #TODO this method was copied from the latest version of HF transformers library to support
    dict outputs. So we should remove it when T4Rec is updated to use the latest version
    """
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        """
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst).
        # Until Numpy adds bfloat16, we must convert float32.
        """
        t = t.to(torch.float32)
    return t.numpy()


def nested_truncate(tensors, limit):
    """Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors).
    #TODO this method was copied from the latest version of HF transformers library to support
    dict outputs. So we should remove it when T4Rec is updated to use the latest version
    """
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_truncate(t, limit) for k, t in tensors.items()})

    return tensors[:limit]


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """
    Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary.
    #TODO this method was copied from the latest version of HF transformers library to support
    dict outputs. So we should remove it when T4Rec is updated to use the latest version
    """
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (
        array1.shape[0] + array2.shape[0],
        max(array1.shape[1], array2.shape[1]),
    ) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def one_hot_1d(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    r"""Coverts a 1d label tensor to one-hot representation

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch size. Each value is an integer
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
