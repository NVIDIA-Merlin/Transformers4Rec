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

import tensorflow as tf

if sys.version_info < (3, 7):
    ForwardRef = typing._ForwardRef  # type: ignore
else:
    ForwardRef = typing.ForwardRef

# TODO: Make this more generic and work with multi-hot features
TabularData = Dict[str, tf.Tensor]
TensorOrTabularData = typing.Union[tf.Tensor, TabularData]

_tabular_module = "transformers4rec.tf.tabular.tabular"
TabularTransformation = ForwardRef(f"{_tabular_module}.TabularTransformation")
TabularAggregation = ForwardRef(f"{_tabular_module}.TabularAggregation")
SequentialTabularTransformations = ForwardRef(f"{_tabular_module}.SequentialTabularTransformations")
TabularTransformationType = ForwardRef(f"{_tabular_module}.TabularTransformationType")
TabularAggregationType = ForwardRef(f"{_tabular_module}.TabularAggregationType")
TabularBlock = ForwardRef(f"{_tabular_module}.TabularBlock")

_features_module = "transformers4rec.tf.features"
TabularFeatures = ForwardRef(f"{_features_module}.tabular.TabularFeatures")
TabularSequenceFeatures = ForwardRef(f"{_features_module}.sequence.TabularSequenceFeatures")
TabularFeaturesType = typing.Union[TabularSequenceFeatures, TabularFeatures]
InputBlock = ForwardRef(f"{_features_module}.base.InputBlock")

FeatureAggregator = ForwardRef("transformers4rec.tf.aggregator.FeatureAggregation")
MaskSequence = ForwardRef("transformers4rec.tf.masking.MaskSequence")

Block = ForwardRef("transformers4rec.tf.block.base.Block")
SequentialBlock = ForwardRef("transformers4rec.tf.block.base.SequentialBlock")
BuildableBlock = ForwardRef("transformers4rec.tf.block.base.BuildableBlock")
BlockWithHead = ForwardRef("transformers4rec.tf.block.with_head.BlockWithHead")

PredictionTask = ForwardRef("transformers4rec.tf.model.prediction_task.PredictionTask")
Head = ForwardRef("transformers4rec.tf.model.head.Head")
Model = ForwardRef("transformers4rec.tf.model.model.Model")

LossReduction = typing.Callable[[typing.List[tf.Tensor]], tf.Tensor]

__all__ = [
    "TabularData",
    "TensorOrTabularData",
    "TabularTransformation",
    "TabularAggregation",
    "SequentialTabularTransformations",
    "TabularTransformationType",
    "TabularAggregationType",
    "TabularBlock",
    "TabularFeatures",
    "TabularSequenceFeatures",
    "TabularFeaturesType",
    "InputBlock",
    "LossReduction",
    "FeatureAggregator",
    "MaskSequence",
    "Block",
    "SequentialBlock",
    "BuildableBlock",
    "BlockWithHead",
    "Head",
    "PredictionTask",
    "Model",
]
