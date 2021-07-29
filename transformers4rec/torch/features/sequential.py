from typing import Dict

import torch

from transformers4rec.torch.utils.torch_utils import calculate_batch_size_from_input_size

from ..block.mlp import MLPBlock
from ..tabular import TabularModule
from .embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .tabular import AsTabular, TabularFeatures


class SequentialEmbeddingFeatures(EmbeddingFeatures):
    def __init__(self, feature_config: Dict[str, FeatureConfig], padding_idx: int = 0, **kwargs):
        self.padding_idx = padding_idx
        super().__init__(feature_config, **kwargs)

    def table_to_embedding_module(self, table: TableConfig) -> torch.nn.Embedding:
        return torch.nn.Embedding(table.vocabulary_size, table.dim, padding_idx=self.padding_idx)

    def forward_output_size(self, input_sizes):
        sizes = {}
        batch_size = calculate_batch_size_from_input_size(input_sizes)
        sequence_length = input_sizes[list(self.feature_config.keys())[0]][1]
        for name, feature in self.feature_config.items():
            sizes[name] = torch.Size([batch_size, sequence_length, feature.table.dim])

        return TabularModule.forward_output_size(self, sizes)


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
