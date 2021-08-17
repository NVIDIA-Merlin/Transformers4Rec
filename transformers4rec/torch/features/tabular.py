from typing import List, Optional, Tuple

from ...types import DatasetSchema, DefaultTags, Tag, Union
from ..block.mlp import MLPBlock
from ..tabular import AsTabular, MergeTabular, TabularModule
from ..utils.torch_utils import get_output_sizes_from_schema
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures, SoftEmbeddingFeatures


class TabularFeatures(MergeTabular):
    CONTINUOUS_MODULE_CLASS = ContinuousFeatures
    EMBEDDING_MODULE_CLASS = EmbeddingFeatures
    SOFT_EMBEDDING_MODULE_CLASS = SoftEmbeddingFeatures

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

        assert to_merge != [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(to_merge, aggregation=aggregation)

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

        continuous = continuous >> MLPBlock(mlp_layers_dims) >> AsTabular("continuous_projection")

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
        continuous_soft_embeddings_shape: Optional[Union[Tuple[int, int], List[int, int]]] = None,
        **kwargs,
    ) -> "TabularFeatures":
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

        Returns
        -------
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            if continuous_soft_embeddings_shape:
                assert (
                    isinstance(continuous_soft_embeddings_shape, (list, tuple))
                    and len(continuous_soft_embeddings_shape) == 2
                ), (
                    "The continuous_soft_embeddings_shape must be a list/tuple with "
                    "2 elements corresponding to the default shape "
                    "for the soft embedding tables of continuous features"
                )

                (
                    default_soft_embedding_cardinality,
                    default_soft_embedding_dim,
                ) = continuous_soft_embeddings_shape
                maybe_continuous_module = cls.SOFT_EMBEDDING_MODULE_CLASS.from_schema(
                    schema,
                    tags=continuous_tags,
                    default_soft_embedding_cardinality=default_soft_embedding_cardinality,
                    default_soft_embedding_dim=default_soft_embedding_dim,
                )
            else:
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

        if output.aggregation is not None:
            output.aggregation.schema = schema

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
