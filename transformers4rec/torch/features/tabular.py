from ...types import ColumnGroup, Tag
from ..block.mlp import MLPBlock
from ..tabular import AsTabular, MergeTabular
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

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "concat"

        continuous = continuous >> MLPBlock(dimensions) >> AsTabular("continuous_projection")

        self.to_merge["continuous_module"] = continuous

        return self

    @classmethod
    def from_column_group(
        cls,
        column_group: ColumnGroup,
        continuous_tags=Tag.CONTINUOUS,
        continuous_tags_to_filter=None,
        categorical_tags=Tag.CATEGORICAL,
        categorical_tags_to_filter=None,
        aggregation=None,
        automatic_build=True,
        max_sequence_length=None,
        continuous_projection=None,
        **kwargs
    ):
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            maybe_continuous_module = cls.CONTINUOUS_MODULE_CLASS.from_column_group(
                column_group, tags=continuous_tags, tags_to_filter=continuous_tags_to_filter
            )
        if categorical_tags:
            maybe_categorical_module = cls.EMBEDDING_MODULE_CLASS.from_column_group(
                column_group, tags=categorical_tags, tags_to_filter=categorical_tags_to_filter
            )

        output = cls(
            continuous_module=maybe_continuous_module,
            categorical_module=maybe_categorical_module,
            text_embedding_module=None,
            aggregation=aggregation,
        )

        if column_group._schema:
            output.build(
                get_output_sizes_from_schema(
                    column_group._schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                )
            )

        if continuous_projection:
            output = output.project_continuous_features(continuous_projection)

        return output

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.to_apply:
            output_sizes.update(in_layer.forward_output_size(input_size))

        return super().forward_output_size(output_sizes)
