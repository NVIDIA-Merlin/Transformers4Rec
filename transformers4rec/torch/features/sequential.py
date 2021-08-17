from typing import Dict, Optional

import torch

from transformers4rec.torch.masking import masking_registry
from transformers4rec.torch.utils.torch_utils import calculate_batch_size_from_input_size

from ...types import DatasetSchema, Tag
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

    def __init__(
        self,
        continuous_module=None,
        categorical_module=None,
        text_embedding_module=None,
        projection_module=None,
        masking=None,
        aggregation=None,
    ):
        super().__init__(continuous_module, categorical_module, text_embedding_module, aggregation)
        if masking:
            self.masking = masking
        self.projection_module = projection_module

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        continuous_tags=Tag.CONTINUOUS,
        categorical_tags=Tag.CATEGORICAL,
        aggregation=None,
        automatic_build=True,
        max_sequence_length=None,
        continuous_projection=None,
        projection=None,
        d_output=None,
        masking=None,
        **kwargs
    ):
        output = super().from_schema(
            schema,
            continuous_tags,
            categorical_tags,
            aggregation,
            automatic_build,
            max_sequence_length,
            continuous_projection,
            **kwargs
        )
        if (projection or masking or d_output) and not aggregation:
            # TODO: print warning here for clarity
            output.aggregation = "sequential_concat"
        hidden_size = output.output_size()

        if d_output and not projection:
            projection = MLPBlock([d_output])
        if projection and hasattr(projection, "build"):
            projection = projection.build(hidden_size)
        if projection:
            output.projection_module = projection
            hidden_size = projection.output_size()

        if isinstance(masking, str):
            masking = masking_registry.parse(masking)(hidden_size=hidden_size[-1], **kwargs)

        output.masking = masking
        # output.hidden_size = hidden_size
        return output

    # @hidden_size.setter
    # def masking(self, value):

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
        if value and not getattr(self.categorical_module, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")

        self._masking = value

    @property
    def item_id(self) -> Optional[str]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_id", None)

        return None

    @property
    def item_embedding_table(self) -> Optional[torch.nn.Module]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_embedding_table", None)

        return None

    def forward(self, inputs, training=True):
        outputs = super(SequentialTabularFeatures, self).forward(inputs)

        if self.masking or self.projection_module:
            outputs = self.aggregation(outputs)

        if self.projection_module:
            outputs = self.projection_module(outputs)

        if self.masking:
            outputs = self.masking(
                outputs, item_ids=self.to_merge["categorical_module"].item_seq, training=training
            )

        return outputs

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "sequential_concat"

        continuous = continuous >> MLPBlock(dimensions) >> AsTabular("continuous_projection")

        self.to_merge["continuous_module"] = continuous

        return self

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.merge_values:
            output_sizes.update(in_layer.forward_output_size(input_size))

        output_sizes = TabularModule.forward_output_size(self, output_sizes)

        if self.projection_module:
            output_sizes = self.projection_module.output_size()

        return output_sizes
