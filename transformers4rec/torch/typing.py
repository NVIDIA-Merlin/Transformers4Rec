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

import sys
import typing
from typing import Dict

import torch

if sys.version_info < (3, 7):
    ForwardRef = typing._ForwardRef  # pylint: disable=protected-access
else:
    ForwardRef = typing.ForwardRef

# TODO: Make this more generic and work with multi-hot features
TabularData = Dict[str, torch.Tensor]
TensorOrTabularData = typing.Union[torch.Tensor, TabularData]

_tabular_module = "transformers4rec.torch.block.tabular.tabular"
TabularTransformation = ForwardRef(f"{_tabular_module}.TabularTransformation")
TabularAggregation = ForwardRef(f"{_tabular_module}.TabularAggregation")
SequentialTabularTransformations = ForwardRef(f"{_tabular_module}.SequentialTabularTransformations")
TabularTransformationType = ForwardRef(f"{_tabular_module}.TabularTransformationType")
TabularAggregationType = ForwardRef(f"{_tabular_module}.TabularAggregationType")
TabularModule = ForwardRef(f"{_tabular_module}.TabularModule")
TabularBlock = ForwardRef(f"{_tabular_module}.TabularBlock")

ProcessedSequence = ForwardRef("transformers4rec.torch.features.ProcessedSequence")

FeatureAggregator = ForwardRef("transformers4rec.torch.aggregator.FeatureAggregation")

MaskSequence = ForwardRef("transformers4rec.torch.masking.MaskSequence")
MaskedSequence = ForwardRef("transformers4rec.torch.masking.MaskedSequence")

Block = ForwardRef("transformers4rec.torch.block.base.Block")
SequentialBlock = ForwardRef("transformers4rec.torch.block.base.SequentialBlock")
BuildableBlock = ForwardRef("transformers4rec.torch.block.base.BuildableBlock")
BlockWithHead = ForwardRef("transformers4rec.torch.block.with_head.BlockWithHead")
BlockType = typing.Union[Block, BuildableBlock]
BlockOrModule = typing.Union[Block, BuildableBlock, torch.nn.Module]

FeatureConfig = ForwardRef("transformers4rec.torch.features.embedding.FeatureConfig")
TableConfig = ForwardRef("transformers4rec.torch.features.embedding.TableConfig")


Head = ForwardRef("transformers4rec.torch.head.Head")
PredictionTask = ForwardRef("transformers4rec.torch.head.PredictionTask")

__all__ = [
    "TabularData",
    "TensorOrTabularData",
    "TabularModule",
    "ProcessedSequence",
    "FeatureAggregator",
    "MaskSequence",
    "MaskedSequence",
    "Block",
    "BlockType",
    "BlockOrModule",
    "SequentialBlock",
    "TableConfig",
    "FeatureConfig",
    "BuildableBlock",
    "Head",
    "PredictionTask",
]
