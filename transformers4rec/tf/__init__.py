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

from .block.base import Block, SequentialBlock, right_shift_layer
from .block.dlrm import DLRMBlock
from .block.mlp import MLPBlock
from .features.continuous import ContinuousFeatures
from .features.embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .features.sequence import SequentialEmbeddingFeatures, TabularSequenceFeatures
from .features.tabular import TabularFeatures
from .features.text import TextEmbeddingFeaturesWithTransformers
from .model.head import BinaryClassificationTask, Head, PredictionTask, RegressionTask
from .model.model import Model
from .tabular.aggregation import (
    ConcatFeatures,
    ElementwiseSum,
    ElementwiseSumItemMulti,
    StackFeatures,
)
from .tabular.tabular import (
    AsDenseFeatures,
    AsSparseFeatures,
    AsTabular,
    FilterFeatures,
    MergeTabular,
    TabularBlock,
)
from .tabular.transformations import StochasticSwapNoise
from .utils import repr_utils

ListWrapper.__repr__ = repr_utils.list_wrapper_repr
_DictWrapper.__repr__ = repr_utils.dict_wrapper_repr

Dense.repr_extra = repr_utils.dense_extra_repr
Layer.__rrshift__ = right_shift_layer
Layer.__repr__ = repr_utils.layer_repr
Loss.__repr__ = repr_utils.layer_repr_no_children
Metric.__repr__ = repr_utils.layer_repr_no_children
OptimizerV2.__repr__ = repr_utils.layer_repr_no_children

__all__ = [
    "Block",
    "SequentialBlock",
    "right_shift_layer",
    "DLRMBlock",
    "MLPBlock",
    "TabularBlock",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "SequentialEmbeddingFeatures",
    "FeatureConfig",
    "TableConfig",
    "TabularFeatures",
    "TabularSequenceFeatures",
    "TextEmbeddingFeaturesWithTransformers",
    "Head",
    "AsDenseFeatures",
    "AsSparseFeatures",
    "ElementwiseSum",
    "ElementwiseSumItemMulti",
    "AsTabular",
    "ConcatFeatures",
    "FilterFeatures",
    "MergeTabular",
    "StackFeatures",
    "PredictionTask",
    "BinaryClassificationTask",
    "RegressionTask",
    "Model",
    "StochasticSwapNoise",
]

try:
    from .data import DataLoader, DataLoaderValidator  # noqa: F401

    __all__.extend(["DataLoader", "DataLoaderValidator"])
except ImportError:
    pass
