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
from tensorflow.keras.layers import Dense, Layer
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from . import repr as _repr
from .blocks.base import Block, SequentialBlock, right_shift_layer
from .blocks.dlrm import DLRMBlock
from .blocks.mlp import MLPBlock
from .blocks.with_head import BlockWithHead
from .features.continuous import ContinuousFeatures
from .features.embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .features.tabular import TabularFeatures
from .features.text import TextEmbeddingFeaturesWithTransformers
from .heads import Head
from .tabular import (
    AsDenseFeatures,
    AsSparseFeatures,
    AsTabular,
    ConcatFeatures,
    FilterFeatures,
    MergeTabular,
    StackFeatures,
    TabularLayer,
)

ListWrapper.__repr__ = _repr.list_wrapper_repr
_DictWrapper.__repr__ = _repr.dict_wrapper_repr

Dense.repr_extra = _repr.dense_extra_repr
Layer.__rrshift__ = right_shift_layer
Layer.__repr__ = _repr.layer_repr
Loss.__repr__ = _repr.layer_repr_no_children
Metric.__repr__ = _repr.layer_repr_no_children
OptimizerV2.__repr__ = _repr.layer_repr_no_children

__all__ = [
    "Block",
    "SequentialBlock",
    "right_shift_layer",
    "DLRMBlock",
    "MLPBlock",
    "BlockWithHead",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "FeatureConfig",
    "TableConfig",
    "TabularFeatures",
    "TextEmbeddingFeaturesWithTransformers",
    "Head",
    "AsDenseFeatures",
    "AsSparseFeatures",
    "AsTabular",
    "ConcatFeatures",
    "FilterFeatures",
    "MergeTabular",
    "StackFeatures",
    "TabularLayer",
]

try:
    from .data import DataLoader, DataLoaderValidator  # noqa: F401

    __all__.extend(["DataLoader", "DataLoaderValidator"])
except ImportError:
    pass
