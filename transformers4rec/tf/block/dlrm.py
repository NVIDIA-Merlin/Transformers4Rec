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

from typing import List, Optional, Union, cast

import tensorflow as tf

from merlin_standard_lib import Schema, Tag

from ..features.continuous import ContinuousFeatures
from ..features.embedding import EmbeddingFeatures
from ..tabular.base import TabularBlock
from .base import Block, BlockType


class ExpandDimsAndToTabular(tf.keras.layers.Lambda):
    def __init__(self, **kwargs):
        super().__init__(lambda x: dict(continuous=x), **kwargs)


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class DLRMBlock(Block):
    def __init__(
        self,
        continuous_features: Union[List[str], Schema, Optional[TabularBlock]],
        embedding_layer: EmbeddingFeatures,
        bottom_mlp: BlockType,
        top_mlp: Optional[BlockType] = None,
        interaction_layer: Optional[tf.keras.layers.Layer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        _continuous_features: Optional[TabularBlock]
        if isinstance(continuous_features, Schema):
            _continuous_features = cast(
                Optional[TabularBlock],
                ContinuousFeatures.from_schema(
                    cast(Schema, continuous_features), aggregation="concat"
                ),
            )
        if isinstance(continuous_features, list):
            _continuous_features = ContinuousFeatures.from_features(
                continuous_features, aggregation="concat"
            )
        else:
            _continuous_features = cast(Optional[TabularBlock], continuous_features)

        if _continuous_features:
            continuous_embedding = _continuous_features >> bottom_mlp >> ExpandDimsAndToTabular()
            continuous_embedding.block_name = "ContinuousEmbedding"
            self.stack_features = embedding_layer.merge(continuous_embedding, aggregation="stack")
        else:
            embedding_layer.set_aggregation("stack")
            self.stack_features = embedding_layer

        # self.stack_features = tabular.MergeTabular(embedding_layer, continuous_embedding,
        #                                            aggregation_registry="stack")

        # self.stack_features = embedding_layer + continuous_embedding
        # self.stack_features.aggregation_registry = "stack"

        from ..layers import DotProductInteraction

        self.interaction_layer = interaction_layer or DotProductInteraction()

        self.top_mlp = top_mlp

    @classmethod
    def from_schema(
        cls, schema: Schema, bottom_mlp: BlockType, top_mlp: Optional[BlockType] = None, **kwargs
    ):
        embedding_layer = EmbeddingFeatures.from_schema(
            schema.select_by_tag(Tag.CATEGORICAL),
            infer_embedding_sizes=False,
            embedding_dim_default=bottom_mlp.layers[-1].units,
        )
        if not embedding_layer:
            raise ValueError("embedding_layer must be set.")

        continuous_features = cast(
            Optional[TabularBlock],
            ContinuousFeatures.from_schema(
                schema.select_by_tag(Tag.CONTINUOUS), aggregation="concat"
            ),
        )

        return cls(continuous_features, embedding_layer, bottom_mlp, top_mlp=top_mlp, **kwargs)

    def call(self, inputs, **kwargs):
        stacked = self.stack_features(inputs)
        interactions = self.interaction_layer(stacked)

        return interactions if not self.top_mlp else self.top_mlp(interactions)
