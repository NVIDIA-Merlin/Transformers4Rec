from typing import List, Optional, Union

from ...types import DatasetSchema, DefaultTags, Tag
from ...utils.misc_utils import docstring_parameter
from .. import typing
from ..block.base import SequentialBlock
from ..block.mlp import MLPBlock
from ..tabular.tabular import TABULAR_MODULE_PARAMS_DOCSTRING, AsTabular, MergeTabular
from ..utils.torch_utils import get_output_sizes_from_schema
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures, SoftEmbeddingFeatures

TABULAR_FEATURES_PARAMS_DOCSTRING = """
    continuous_module: TabularModule, optional
        Module used to process continuous features.
    categorical_module: TabularModule, optional
        Module used to process categorical features.
    text_embedding_module: TabularModule, optional
        Module used to process text features.
"""


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
class TabularFeatures(MergeTabular):
    """Input module that combines different types of features: continuous, categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    {tabular_module_parameters}
    """

    CONTINUOUS_MODULE_CLASS = ContinuousFeatures
    EMBEDDING_MODULE_CLASS = EmbeddingFeatures
    SOFT_EMBEDDING_MODULE_CLASS = SoftEmbeddingFeatures

    def __init__(
        self,
        continuous_module: Optional[typing.TabularModule] = None,
        categorical_module: Optional[typing.TabularModule] = None,
        text_embedding_module: Optional[typing.TabularModule] = None,
        pre: Optional[typing.TabularTransformationType] = None,
        post: Optional[typing.TabularTransformationType] = None,
        aggregation: Optional[typing.TabularAggregationType] = None,
    ):
        to_merge = {}
        if continuous_module:
            to_merge["continuous_module"] = continuous_module
        if categorical_module:
            to_merge["categorical_module"] = categorical_module
        if text_embedding_module:
            to_merge["text_embedding_module"] = text_embedding_module

        assert to_merge != [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(to_merge, pre=pre, post=post, aggregation=aggregation)

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

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "concat"

        continuous = SequentialBlock(
            continuous, MLPBlock(mlp_layers_dims), AsTabular("continuous_projection")
        )

        self.to_merge["continuous_module"] = continuous

        return self

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        continuous_tags: Optional[Union[DefaultTags, list, str]] = Tag.CONTINUOUS,
        categorical_tags: Optional[Union[DefaultTags, list, str]] = Tag.CATEGORICAL,
        aggregation: Optional[str] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        continuous_soft_embeddings: bool = False,
        **kwargs,
    ) -> "TabularFeatures":
        """Instantiates ``TabularFeatures`` from a ``DatasetSchema``

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
            If set, concatenate all numerical features and project them by a number of MLP layers.
            The argument accepts a list with the dimensions of the MLP layers, by default None
        continuous_soft_embeddings : bool
            Indicates if the  soft one-hot encoding technique must be used to
            represent continuous features, by default False

        Returns
        -------
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            if continuous_soft_embeddings:
                maybe_continuous_module = cls.SOFT_EMBEDDING_MODULE_CLASS.from_schema(
                    schema,
                    tags=continuous_tags,
                    **kwargs,
                )
            else:
                maybe_continuous_module = cls.CONTINUOUS_MODULE_CLASS.from_schema(
                    schema, tags=continuous_tags
                )
        if categorical_tags:
            maybe_categorical_module = cls.EMBEDDING_MODULE_CLASS.from_schema(
                schema, tags=categorical_tags, **kwargs
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
                ),
                schema=schema,
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

        return output_sizes

    @property
    def continuous_module(self) -> Optional[typing.TabularModule]:
        if "continuous_module" in self.to_merge:
            return self.to_merge["continuous_module"]

        return None

    @property
    def categorical_module(self) -> Optional[typing.TabularModule]:
        if "categorical_module" in self.to_merge:
            return self.to_merge["categorical_module"]

        return None
