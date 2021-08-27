from typing import List, Optional, Union

import tensorflow as tf

from ...types import DatasetSchema
from ...utils.schema import Tag
from .. import tabular
from ..features.continuous import ContinuousFeatures
from ..features.embedding import EmbeddingFeatures
from .base import Block, BlockType


class ExpandDimsAndToTabular(tf.keras.layers.Lambda):
    def __init__(self, **kwargs):
        super().__init__(lambda x: dict(continuous=tf.expand_dims(x, 1)), **kwargs)


class DLRMBlock(Block):
    def __init__(
        self,
        continuous_features: Union[List[str], DatasetSchema, tabular.TabularLayer],
        embedding_layer: EmbeddingFeatures,
        bottom_mlp: BlockType,
        top_mlp: Optional[BlockType] = None,
        interaction_layer: Optional[tf.keras.layers.Layer] = None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        if isinstance(continuous_features, DatasetSchema):
            continuous_features = ContinuousFeatures.from_schema(
                continuous_features, aggregation="concat"
            )
        if isinstance(continuous_features, list):
            continuous_features = ContinuousFeatures.from_features(
                continuous_features, aggregation="concat"
            )

        continuous_embedding = continuous_features >> bottom_mlp >> ExpandDimsAndToTabular()
        continuous_embedding.block_name = "ContinuousEmbedding"

        # self.stack_features = tabular.MergeTabular(embedding_layer, continuous_embedding,
        #                                            aggregation_registry="stack")

        # self.stack_features = embedding_layer + continuous_embedding
        # self.stack_features.aggregation_registry = "stack"

        self.stack_features = embedding_layer.merge(continuous_embedding, aggregation="stack")

        from ..layers import DotProductInteraction

        self.interaction_layer = interaction_layer or DotProductInteraction()

        self.top_mlp = top_mlp

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        bottom_mlp: BlockType,
        top_mlp: Optional[BlockType] = None,
        **kwargs
    ):
        embedding_layer = EmbeddingFeatures.from_schema(
            schema.select_by_tag(Tag.CATEGORICAL),
            infer_embedding_sizes=False,
            embedding_dim_default=bottom_mlp.layers[-1].units,
        )

        continuous_features = ContinuousFeatures.from_schema(
            schema.select_by_tag(Tag.CONTINUOUS), aggregation="concat"
        )

        return cls(continuous_features, embedding_layer, bottom_mlp, top_mlp=top_mlp, **kwargs)

    def call(self, inputs, **kwargs):
        stacked = self.stack_features(inputs)
        interactions = self.interaction_layer(stacked)

        return interactions if not self.top_mlp else self.top_mlp(interactions)
