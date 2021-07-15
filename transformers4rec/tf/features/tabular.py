from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures
from .text import TextEmbeddingFeaturesWithTransformers
from ..tabular import MergeTabular
from ...types import ColumnGroup, Tag


class TabularFeatures(MergeTabular):
    def __init__(
        self,
        continuous_layer=None,
        categorical_layer=None,
        text_embedding_layer=None,
        aggregation=None,
        **kwargs
    ):
        to_merge = []
        if continuous_layer:
            to_merge.append(continuous_layer)
        if categorical_layer:
            to_merge.append(categorical_layer)
        if text_embedding_layer:
            to_merge.append(text_embedding_layer)

        assert to_merge is not [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(*to_merge, aggregation=aggregation, **kwargs)

    @classmethod
    def from_column_group(
        cls,
        column_group: ColumnGroup,
        continuous_tags=Tag.CONTINUOUS,
        continuous_tags_to_filter=None,
        categorical_tags=Tag.CATEGORICAL,
        categorical_tags_to_filter=None,
        text_model=None,
        text_tags=Tag.TEXT_TOKENIZED,
        text_tags_to_filter=None,
        max_text_length=None,
        aggregation=None,
        **kwargs
    ):
        maybe_continuous_layer, maybe_categorical_layer = None, None
        if continuous_tags:
            maybe_continuous_layer = ContinuousFeatures.from_column_group(
                column_group, tags=continuous_tags, tags_to_filter=continuous_tags_to_filter
            )
        if categorical_tags:
            maybe_categorical_layer = EmbeddingFeatures.from_column_group(
                column_group, tags=categorical_tags, tags_to_filter=categorical_tags_to_filter
            )

        if text_model and not isinstance(text_model, TextEmbeddingFeaturesWithTransformers):
            text_model = TextEmbeddingFeaturesWithTransformers.from_column_group(
                column_group,
                tags=text_tags,
                tags_to_filter=text_tags_to_filter,
                transformer_model=text_model,
                max_text_length=max_text_length,
            )

        return cls(
            continuous_layer=maybe_continuous_layer,
            categorical_layer=maybe_categorical_layer,
            text_embedding_layer=text_model,
            aggregation=aggregation,
            **kwargs
        )
