from typing import Dict, Optional

import torch

from transformers4rec.torch.masking import masking_registry
from transformers4rec.torch.utils.torch_utils import calculate_batch_size_from_input_size

from ...types import DatasetSchema, Tag
from ..block.base import SequentialBlock
from ..block.mlp import MLPBlock
from ..tabular import TabularModule
from .embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .tabular import AsTabular, TabularFeatures


class SequenceEmbeddingFeatures(EmbeddingFeatures):
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


class TabularTransformerFeatures(TabularFeatures):
    EMBEDDING_MODULE_CLASS = SequenceEmbeddingFeatures

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
        continuous_soft_embeddings_shape=None,
        projection=None,
        d_output=None,
        masking=None,
        **kwargs
    ) -> "TabularTransformerFeatures":
        """Instantiates ``TabularFeatures`` from a ```DatasetSchema`
        Parameters
        ----------
        schema : DatasetSchema
            Dataset schema
        continuous_tags : Optional[Union[DefaultTags, list, str]], optional
            Tags to filter the continuous features, by default Tag.CONTINUOUS
        categorical_tags : Optional[Union[DefaultTags, list, str]], optional
            Tags to filter the categorical features, by default Tag.CATEGORICAL
        aggregation : Optional[str], optional
            Feature aggregation option, by default None
        automatic_build : bool, optional
            Automatically infers input size from features, by default True
        max_sequence_length : Optional[int], optional
            Maximum sequence length for list features by default None
        continuous_projection : Optional[Union[List[int], int]], optional
            If set, concatenate all numerical features and projet them by a number of MLP layers.
            The argument accepts a list with the dimensions of the MLP layers, by default None
        continuous_soft_embeddings_shape : Optional[Union[Tuple[int, int], List[int, int]]]
            If set, uses soft one-hot encoding technique to represent continuous features.
            The argument accepts a tuple with 2 elements: [embeddings cardinality, embeddings dim],
            by default None
        projection: Optional[torch.nn.Module, BuildableBlock], optional
            If set, project the aggregated embeddings vectors into hidden dimension vector space,
            by default None
        d_output: Optional[int], optional
            If set, init a MLPBlock as projection module to project embeddings vectors,
            by default None
        masking: Optional[Union[str, MaskSequence]], optional
            If set, Apply masking to the input embeddings and compute masked labels, It requires
            a categorical_module including an item_id column, by default None

        Returns
        -------
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        output = super().from_schema(
            schema,
            continuous_tags,
            categorical_tags,
            aggregation,
            automatic_build,
            max_sequence_length,
            continuous_projection,
            continuous_soft_embeddings_shape,
            **kwargs
        )
        if d_output and projection:
            raise ValueError("You cannot specify both d_output and projection at the same time")
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
        if masking and not getattr(output, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")
        output.masking = masking

        return output

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
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
        outputs = super(TabularTransformerFeatures, self).forward(inputs)

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

        continuous = SequentialBlock(
            continuous, MLPBlock(dimensions), AsTabular("continuous_projection")
        )

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
