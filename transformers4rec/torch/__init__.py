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

from .aggregation import ConcatFeatures, ElementwiseSum, ElementwiseSumItemMulti, StackFeatures
from .augmentation import StochasticSwapNoise
from .block.base import Block, SequentialBlock, build_blocks, right_shift_block
from .block.mlp import MLPBlock
from .block.transformer import TransformerBlock
from .features.continuous import ContinuousFeatures
from .features.embedding import (
    EmbeddingFeatures,
    FeatureConfig,
    SoftEmbedding,
    SoftEmbeddingFeatures,
    TableConfig,
)
from .features.sequential import SequentialEmbeddingFeatures, SequentialTabularFeatures
from .features.tabular import TabularFeatures
from .head import (
    BinaryClassificationTask,
    Head,
    NextItemPredictionTask,
    PredictionTask,
    RegressionTask,
)
from .model import Model
from .tabular import AsTabular, FilterFeatures, MergeTabular, TabularModule

__all__ = [
    "SequentialBlock",
    "right_shift_block",
    "build_blocks",
    "Block",
    "MLPBlock",
    "StochasticSwapNoise",
    "TransformerBlock",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "SoftEmbeddingFeatures",
    "SequentialTabularFeatures",
    "SequentialEmbeddingFeatures",
    "FeatureConfig",
    "TableConfig",
    "TabularFeatures",
    "Head",
    "Model",
    "PredictionTask",
    "AsTabular",
    "ConcatFeatures",
    "FilterFeatures",
    "ElementwiseSum",
    "ElementwiseSumItemMulti",
    "MergeTabular",
    "StackFeatures",
    "BinaryClassificationTask",
    "RegressionTask",
    "NextItemPredictionTask",
    "TabularModule",
    "SoftEmbedding",
]

try:
    from .data import DataLoader, DataLoaderValidator  # noqa: F401

    __all__.append("DataLoader")
except ImportError:
    pass
