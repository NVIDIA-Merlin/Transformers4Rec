from typing import List

from ...types import Schema, Tag
from ..block.mlp import MLPBlock
from ..tabular import AsTabular, MergeTabular, TabularModule
from ..utils.torch_utils import get_output_sizes_from_schema
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures


class TabularFeatures(MergeTabular):
    CONTINUOUS_MODULE_CLASS = ContinuousFeatures
    EMBEDDING_MODULE_CLASS = EmbeddingFeatures

    def __init__(
        self,
        continuous_module=None,
        categorical_module=None,
        text_embedding_module=None,
        aggregation=None,
    ):
        to_merge = {}
        if continuous_module:
            to_merge["continuous_module"] = continuous_module
        if categorical_module:
            to_merge["categorical_module"] = categorical_module
        if text_embedding_module:
            to_merge["text_embedding_module"] = text_embedding_module

        assert to_merge is not [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(to_merge, aggregation=aggregation)

    def project_continuous_features(self, dimensions: List[int]):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "concat"

        continuous = continuous >> MLPBlock(dimensions) >> AsTabular("continuous_projection")

        self.to_merge["continuous_module"] = continuous

        return self

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        continuous_tags=Tag.CONTINUOUS,
        categorical_tags=Tag.CATEGORICAL,
        aggregation=None,
        automatic_build=True,
        max_sequence_length=None,
        continuous_projection=None,
        **kwargs
    ):
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            maybe_continuous_module = cls.CONTINUOUS_MODULE_CLASS.from_schema(
                schema,
                tags=continuous_tags,
            )
        if categorical_tags:
            maybe_categorical_module = cls.EMBEDDING_MODULE_CLASS.from_schema(
                schema,
                tags=categorical_tags,
            )

        output = cls(
            continuous_module=maybe_continuous_module,
            categorical_module=maybe_categorical_module,
            text_embedding_module=None,
            aggregation=aggregation,
        )

        if automatic_build and schema._schema:
            output.build(
                get_output_sizes_from_schema(
                    schema._schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                )
            )

        if continuous_projection:
            if not automatic_build:
                raise ValueError(
                    "Continuous feature projection can only be done with automatic_build"
                )
            output = output.project_continuous_features(continuous_projection)

        return output

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.merge_values:
            output_sizes.update(in_layer.forward_output_size(input_size))

        return TabularModule.forward_output_size(self, output_sizes)
