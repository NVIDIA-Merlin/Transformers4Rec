from ...types import DatasetSchema, Tag
from ..tabular import MergeTabular
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures
from .text import TextEmbeddingFeaturesWithTransformers


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

        assert to_merge != [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(*to_merge, aggregation=aggregation, **kwargs)

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        continuous_tags=Tag.CONTINUOUS,
        categorical_tags=Tag.CATEGORICAL,
        text_model=None,
        text_tags=Tag.TEXT_TOKENIZED,
        max_text_length=None,
        aggregation=None,
        **kwargs
    ):
        maybe_continuous_layer, maybe_categorical_layer = None, None
        if continuous_tags:
            maybe_continuous_layer = ContinuousFeatures.from_schema(
                schema,
                tags=continuous_tags,
            )
        if categorical_tags:
            maybe_categorical_layer = EmbeddingFeatures.from_schema(
                schema,
                tags=categorical_tags,
            )

        if text_model and not isinstance(text_model, TextEmbeddingFeaturesWithTransformers):
            text_model = TextEmbeddingFeaturesWithTransformers.from_schema(
                schema,
                tags=text_tags,
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
