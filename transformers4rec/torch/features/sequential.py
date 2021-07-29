from typing import Dict

import torch

from ..block.mlp import MLPBlock
from .embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .tabular import AsTabular, TabularFeatures


class SequentialEmbeddingFeatures(EmbeddingFeatures):
    def __init__(self, feature_config: Dict[str, FeatureConfig], padding_idx: int = 0, **kwargs):
        self.padding_idx = padding_idx
        super().__init__(feature_config, **kwargs)

    def table_to_embedding_module(self, table: TableConfig) -> torch.nn.Embedding:
        return torch.nn.Embedding(table.vocabulary_size, table.dim, padding_idx=self.padding_idx)


class SequentialTabularFeatures(TabularFeatures):
    EMBEDDING_MODULE_CLASS = SequentialEmbeddingFeatures

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "sequential_concat"

        continuous = continuous >> MLPBlock(dimensions) >> AsTabular("continuous_projection")

        self.to_merge["continuous_module"] = continuous

        return self
