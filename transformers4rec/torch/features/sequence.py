from typing import Dict, List, Optional, Union

import torch

from ...types import DatasetSchema, DefaultTags, Tag
from ...utils.masking import MaskSequence
from ...utils.misc_utils import docstring_parameter
from .. import typing
from ..block.base import BuildableBlock, SequentialBlock
from ..block.mlp import MLPBlock
from ..block.tabular.tabular import TABULAR_MODULE_PARAMS_DOCSTRING, AsTabular
from ..masking import masking_registry
from ..utils.torch_utils import calculate_batch_size_from_input_size
from . import embedding
from .tabular import TABULAR_FEATURES_PARAMS_DOCSTRING, TabularFeatures


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=embedding.EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
class SequenceEmbeddingFeatures(embedding.EmbeddingFeatures):
    """Input block for embedding-lookups for categorical features. This module produces 3-D tensors,
    this is useful for sequential models like transformers.

    Parameters
    ----------
    {embedding_features_parameters}
    padding_idx: int
        The symbol to use for padding.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        feature_config: Dict[str, embedding.FeatureConfig],
        item_id: Optional[str] = None,
        padding_idx: int = 0,
        pre: Optional[typing.TabularTransformationType] = None,
        post: Optional[typing.TabularTransformationType] = None,
        aggregation: Optional[typing.TabularAggregationType] = None,
    ):
        super(SequenceEmbeddingFeatures, self).__init__(
            feature_config=feature_config,
            item_id=item_id,
            pre=pre,
            post=post,
            aggregation=aggregation,
        )
        self.padding_idx = padding_idx

    def table_to_embedding_module(self, table: embedding.TableConfig) -> torch.nn.Embedding:
        embedding_table = torch.nn.Embedding(
            table.vocabulary_size, table.dim, padding_idx=self.padding_idx
        )
        if table.initializer is not None:
            table.initializer(embedding_table.weight)
        return embedding_table

    def forward_output_size(self, input_sizes):
        sizes = {}
        batch_size = calculate_batch_size_from_input_size(input_sizes)
        sequence_length = input_sizes[list(self.feature_config.keys())[0]][1]
        for name, feature in self.feature_config.items():
            sizes[name] = torch.Size([batch_size, sequence_length, feature.table.dim])

        return sizes


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
class TabularSequenceFeatures(TabularFeatures):
    """Input module that combines different types of features to a sequence: continuous,
    categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    projection_module: BlockOrModule, optional
        Module that's used to project the output of this module, typically done by an MLPBlock.
    masking: MaskSequence, optional
         Masking to apply to the inputs.
    {tabular_module_parameters}

    """

    EMBEDDING_MODULE_CLASS = SequenceEmbeddingFeatures

    def __init__(
        self,
        continuous_module: Optional[typing.TabularModule] = None,
        categorical_module: Optional[typing.TabularModule] = None,
        text_embedding_module: Optional[typing.TabularModule] = None,
        projection_module: Optional[typing.BlockOrModule] = None,
        masking: Optional[typing.MaskSequence] = None,
        pre: Optional[typing.TabularTransformationType] = None,
        post: Optional[typing.TabularTransformationType] = None,
        aggregation: Optional[typing.TabularAggregationType] = None,
    ):
        super().__init__(
            continuous_module,
            categorical_module,
            text_embedding_module,
            pre=pre,
            post=post,
            aggregation=aggregation,
        )
        if masking:
            self.masking = masking
        self.projection_module = projection_module

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
        projection: Optional[Union[torch.nn.Module, BuildableBlock]] = None,
        d_output: Optional[int] = None,
        masking: Optional[Union[str, MaskSequence]] = None,
        **kwargs
    ) -> "TabularSequenceFeatures":
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
        continuous_soft_embeddings : bool
            Indicates if the  soft one-hot encoding technique must be used to represent
            continuous features, by default False
        projection: Optional[Union[torch.nn.Module, BuildableBlock]], optional
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
            schema=schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            aggregation=aggregation,
            automatic_build=automatic_build,
            max_sequence_length=max_sequence_length,
            continuous_projection=continuous_projection,
            continuous_soft_embeddings=continuous_soft_embeddings,
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
        outputs = super(TabularSequenceFeatures, self).forward(inputs)

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

        output_sizes = self._check_post_output_size(output_sizes)

        if self.projection_module:
            output_sizes = self.projection_module.output_size()

        return output_sizes
