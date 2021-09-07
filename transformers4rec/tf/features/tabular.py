from typing import List, Optional, Union

import tensorflow as tf

from ...types import DatasetSchema, DefaultTags, Tag
from ...utils.misc_utils import docstring_parameter
from ..block.base import SequentialBlock
from ..block.mlp import MLPBlock
from ..tabular.tabular import TABULAR_MODULE_PARAMS_DOCSTRING, AsTabular, MergeTabular
from ..typing import TabularAggregationType, TabularBlock, TabularTransformationType
from .base import InputBlock
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures
from .text import TextEmbeddingFeaturesWithTransformers

TABULAR_FEATURES_PARAMS_DOCSTRING = """
    continuous_layer: TabularBlock, optional
        Block used to process continuous features.
    categorical_layer: TabularBlock, optional
        Block used to process categorical features.
    text_embedding_layer: TabularBlock, optional
        Block used to process text features.
"""


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class TabularFeatures(InputBlock, MergeTabular):
    """Input block that combines different types of features: continuous, categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    {tabular_module_parameters}
    """

    CONTINUOUS_MODULE_CLASS = ContinuousFeatures
    EMBEDDING_MODULE_CLASS = EmbeddingFeatures

    def __init__(
        self,
        continuous_layer: Optional[TabularBlock] = None,
        categorical_layer: Optional[TabularBlock] = None,
        text_embedding_layer: Optional[TabularBlock] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[DatasetSchema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        to_merge = {}
        if continuous_layer:
            to_merge["continuous_layer"] = continuous_layer
        if categorical_layer:
            to_merge["categorical_layer"] = categorical_layer
        if text_embedding_layer:
            to_merge["text_embedding_layer"] = text_embedding_layer

        assert to_merge != [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(
            to_merge,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            **kwargs
        )

        if continuous_projection:
            self.project_continuous_features(continuous_projection)

    def project_continuous_features(
        self, mlp_layers_dims: Union[List[int], int]
    ) -> "TabularFeatures":
        """Combine all concatenated continuous features with stacked MLP layers

        Parameters
        ----------
        mlp_layers_dims : Union[List[int], int]
            The MLP layer dimensions

        Returns
        -------
        TabularFeatures
            Returns the same ``TabularFeatures`` object with the continuous features projected
        """
        if isinstance(mlp_layers_dims, int):
            mlp_layers_dims = [mlp_layers_dims]

        continuous = self.continuous_layer
        continuous.set_aggregation("concat")

        continuous = SequentialBlock(
            [continuous, MLPBlock(mlp_layers_dims), AsTabular("continuous_projection")]
        )

        self.to_merge["continuous_layer"] = continuous

        return self

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        continuous_tags: Optional[Union[DefaultTags, list, str]] = Tag.CONTINUOUS,
        categorical_tags: Optional[Union[DefaultTags, list, str]] = Tag.CATEGORICAL,
        aggregation: Optional[str] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        text_model=None,
        text_tags=Tag.TEXT_TOKENIZED,
        max_sequence_length=None,
        max_text_length=None,
        **kwargs
    ):
        maybe_continuous_layer, maybe_categorical_layer = None, None
        if continuous_tags:
            maybe_continuous_layer = cls.CONTINUOUS_MODULE_CLASS.from_schema(
                schema,
                tags=continuous_tags,
            )
        if categorical_tags:
            maybe_categorical_layer = cls.EMBEDDING_MODULE_CLASS.from_schema(
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

        output = cls(
            continuous_layer=maybe_continuous_layer,
            categorical_layer=maybe_categorical_layer,
            text_embedding_layer=text_model,
            aggregation=aggregation,
            continuous_projection=continuous_projection,
            schema=schema,
            **kwargs
        )

        return output

    @property
    def continuous_layer(self):
        if "continuous_layer" in self.to_merge:
            return self.to_merge["continuous_layer"]

        return None

    @property
    def categorical_layer(self):
        if "categorical_layer" in self.to_merge:
            return self.to_merge["categorical_layer"]

        return None
