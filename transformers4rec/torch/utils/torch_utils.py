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

import torch

from merlin_standard_lib import Schema
from merlin_standard_lib.utils.proto_utils import has_field
from transformers4rec.config.schema import SchemaMixin


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


def check_gpu(module):
    try:
        return next(module.parameters()).is_cuda
    except StopIteration:
        return False


def get_output_sizes_from_schema(schema: Schema, batch_size=-1, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
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
            sizes[name] = torch.Size([batch_size, 1])

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
        scores.to(dtype=torch.float32, device=scores.device),
        labels.to(dtype=torch.float32, device=scores.device),
    )


def extract_topk(ks, scores, labels):
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 1, topk_indices)
    return topk_scores, topk_indices, topk_labels


def create_output_placeholder(scores, ks):
    return torch.zeros(scores.shape[0], len(ks)).to(device=scores.device, dtype=torch.float32)


def tranform_label_to_onehot(labels, vocab_size):
    return torch.nn.functional.one_hot(labels.reshape(-1), vocab_size).detach()
