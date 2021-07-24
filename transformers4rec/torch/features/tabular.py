from typing import Dict, Union

from ...types import ColumnGroup, Tag
from ..tabular import MergeTabular
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures


class TabularFeatures(MergeTabular):
    def __init__(
        self,
        continuous_module=None,
        categorical_module=None,
        text_embedding_module=None,
        aggregation=None,
    ):
        to_merge = []
        if continuous_module:
            to_merge.append(continuous_module)
        if categorical_module:
            to_merge.append(categorical_module)
        if text_embedding_module:
            to_merge.append(text_embedding_module)

        assert to_merge is not [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(*to_merge, aggregation=aggregation)

    @classmethod
    def from_column_group(
        cls,
        column_group: ColumnGroup,
        continuous_tags=[],
        continuous_tags_to_filter=Tag.CONTINUOUS,
        categorical_tags=Tag.CATEGORICAL,
        categorical_tags_to_filter=None,
        text_model=None,
        text_tags=Tag.TEXT_TOKENIZED,
        text_tags_to_filter=None,
        max_text_length=None,
        aggregation=None,
        **kwargs
    ):
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            maybe_continuous_module = ContinuousFeatures.from_column_group(
                column_group, tags=continuous_tags, tags_to_filter=continuous_tags_to_filter
            )
        if categorical_tags:
            maybe_categorical_module = EmbeddingFeatures.from_column_group(
                column_group, tags=categorical_tags, tags_to_filter=categorical_tags_to_filter
            )

        # if text_model and not isinstance(text_model, TransformersTextEmbedding):
        #     text_model = TransformersTextEmbedding.from_column_group(
        #         column_group,
        #         tags=text_tags,
        #         tags_to_filter=text_tags_to_filter,
        #         transformer_model=text_model,
        #         max_text_length=max_text_length)

        return cls(
            continuous_module=maybe_continuous_module,
            categorical_module=maybe_categorical_module,
            text_embedding_module=text_model,
            aggregation=aggregation,
        )

    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, dict], str],
        continuous_tags=[],
        continuous_tags_to_filter=Tag.CONTINUOUS,
        categorical_tags=Tag.CATEGORICAL,
        categorical_tags_to_filter=None,
        text_model=None,
        text_tags=Tag.TEXT_TOKENIZED,
        text_tags_to_filter=None,
        max_text_length=None,
        aggregation=None,
        **kwargs
    ):
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            maybe_continuous_module = ContinuousFeatures.from_config(
                config, tags=continuous_tags, tags_to_filter=continuous_tags_to_filter
            )
        if categorical_tags:
            maybe_categorical_module = EmbeddingFeatures.from_config(
                config, tags=categorical_tags, tags_to_filter=categorical_tags_to_filter
            )

        return cls(
            continuous_module=maybe_continuous_module,
            categorical_module=maybe_categorical_module,
            text_embedding_module=text_model,
            aggregation=aggregation,
        )

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.to_apply:
            output_sizes.update(in_layer.forward_output_size(input_size))

        return super().forward_output_size(output_sizes)
