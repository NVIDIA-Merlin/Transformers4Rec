#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import partial
from typing import Any, Callable, Dict, Optional, Text, Union

import torch

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.doc_utils import docstring_parameter
from merlin_standard_lib.utils.embedding_utils import get_embedding_sizes_from_schema

from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    FilterFeatures,
    TabularAggregationType,
    TabularTransformationType,
)
from ..utils.torch_utils import calculate_batch_size_from_input_size, get_output_sizes_from_schema
from .base import InputBlock

EMBEDDING_FEATURES_PARAMS_DOCSTRING = """
    feature_config: Dict[str, FeatureConfig]
        This specifies what TableConfig to use for each feature. For shared embeddings, the same
        TableConfig can be used for multiple features.
    item_id: str, optional
        The name of the feature that's used for the item_id.
"""


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
class EmbeddingFeatures(InputBlock):
    """Input block for embedding-lookups for categorical features.

    For multi-hot features, the embeddings will be aggregated into a single tensor using the mean.

    Parameters
    ----------
    {embedding_features_parameters}
    {tabular_module_parameters}
    """

    def __init__(
        self,
        feature_config: Dict[str, "FeatureConfig"],
        item_id: Optional[str] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
    ):
        super().__init__(pre=pre, post=post, aggregation=aggregation, schema=schema)
        self.item_id = item_id
        self.feature_config = feature_config
        self.filter_features = FilterFeatures(list(feature_config.keys()))

        embedding_tables = {}
        features_dim = {}
        tables: Dict[str, TableConfig] = {}
        for name, feature in self.feature_config.items():
            table: TableConfig = feature.table
            features_dim[name] = table.dim
            if name not in tables:
                tables[name] = table

        for name, table in tables.items():
            embedding_tables[name] = self.table_to_embedding_module(table)

        self.embedding_tables = torch.nn.ModuleDict(embedding_tables)

    @property
    def item_embedding_table(self):
        assert self.item_id is not None

        return self.embedding_tables[self.item_id]

    def table_to_embedding_module(self, table: "TableConfig") -> torch.nn.Module:
        embedding_table = EmbeddingBagWrapper(table.vocabulary_size, table.dim, mode=table.combiner)

        if table.initializer is not None:
            table.initializer(embedding_table.weight)
        return embedding_table

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        embedding_dims: Optional[Dict[str, int]] = None,
        embedding_dim_default: int = 64,
        infer_embedding_sizes: bool = False,
        infer_embedding_sizes_multiplier: float = 2.0,
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
        combiner: str = "mean",
        tags: Optional[Union[Tag, list, str]] = None,
        item_id: Optional[str] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
        aggregation=None,
        pre=None,
        post=None,
        **kwargs,
    ) -> Optional["EmbeddingFeatures"]:
        """Instantitates ``EmbeddingFeatures`` from a ``DatasetSchema``.

        Parameters
        ----------
        schema : DatasetSchema
            Dataset schema
        embedding_dims : Optional[Dict[str, int]], optional
            The dimension of the embedding table for each feature (key),
            by default None by default None
        default_embedding_dim : Optional[int], optional
            Default dimension of the embedding table, when the feature is not found
            in ``default_soft_embedding_dim``, by default 64
        infer_embedding_sizes : bool, optional
            Automatically defines the embedding dimension from the
            feature cardinality in the schema,
            by default False
        infer_embedding_sizes_multiplier: Optional[int], by default 2.0
            multiplier used by the heuristic to infer the embedding dimension from
            its cardinality. Generally reasonable values range between 2.0 and 10.0
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]]
            Dict where keys are feature names and values are callable to initialize embedding tables
        combiner : Optional[str], optional
            Feature aggregation option, by default "mean"
        tags : Optional[Union[DefaultTags, list, str]], optional
            Tags to filter columns, by default None
        item_id : Optional[str], optional
            Name of the item id column (feature), by default None
        automatic_build : bool, optional
            Automatically infers input size from features, by default True
        max_sequence_length : Optional[int], optional
            Maximum sequence length for list features,, by default None

        Returns
        -------
        Optional[EmbeddingFeatures]
            Returns the ``EmbeddingFeatures`` for the dataset schema
        """
        # TODO: propagate item-id from ITEM_ID tag

        if tags:
            schema = schema.select_by_tag(tags)

        if not item_id and schema.select_by_tag(["item_id"]).column_names:
            item_id = schema.select_by_tag(["item_id"]).column_names[0]

        embedding_dims = embedding_dims or {}

        if infer_embedding_sizes:
            embedding_dims_infered = get_embedding_sizes_from_schema(
                schema, infer_embedding_sizes_multiplier
            )

            embedding_dims = {
                **embedding_dims,
                **{k: v for k, v in embedding_dims_infered.items() if k not in embedding_dims},
            }

        embeddings_initializers = embeddings_initializers or {}

        emb_config = {}
        cardinalities = schema.categorical_cardinalities()
        for key, cardinality in cardinalities.items():
            embedding_size = embedding_dims.get(key, embedding_dim_default)
            embedding_initializer = embeddings_initializers.get(key, None)
            emb_config[key] = (cardinality, embedding_size, embedding_initializer)

        feature_config: Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim, emb_initilizer) in emb_config.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=name,
                    combiner=combiner,
                    initializer=emb_initilizer,
                )
            )

        if not feature_config:
            return None

        output = cls(feature_config, item_id=item_id, pre=pre, post=post, aggregation=aggregation)

        if automatic_build and schema:
            output.build(
                get_output_sizes_from_schema(
                    schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                ),
                schema=schema,
            )

        return output

    def item_ids(self, inputs) -> torch.Tensor:
        return inputs[self.item_id]

    def forward(self, inputs, **kwargs):
        embedded_outputs = {}
        filtered_inputs = self.filter_features(inputs)
        for name, val in filtered_inputs.items():
            if isinstance(val, tuple):
                values, offsets = val
                values = torch.squeeze(values, -1)
                # for the case where only one value in values
                if len(values.shape) == 0:
                    values = values.unsqueeze(0)
                embedded_outputs[name] = self.embedding_tables[name](values, offsets[:, 0])
            else:
                # if len(val.shape) <= 1:
                #    val = val.unsqueeze(0)
                embedded_outputs[name] = self.embedding_tables[name](val)

        # Store raw item ids for masking and/or negative sampling
        # This makes this module stateful.
        if self.item_id:
            self.item_seq = self.item_ids(inputs)

        embedded_outputs = super().forward(embedded_outputs)

        return embedded_outputs

    def forward_output_size(self, input_sizes):
        sizes = {}
        batch_size = calculate_batch_size_from_input_size(input_sizes)
        for name, feature in self.feature_config.items():
            sizes[name] = torch.Size([batch_size, feature.table.dim])

        return sizes


class EmbeddingBagWrapper(torch.nn.EmbeddingBag):
    def forward(self, input, **kwargs):
        # EmbeddingBag requires 2D tensors (or offsets)
        if len(input.shape) == 1:
            input = input.unsqueeze(-1)
        return super().forward(input, **kwargs)


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
class SoftEmbeddingFeatures(EmbeddingFeatures):
    """
    Encapsulate continuous features encoded using the Soft-one hot encoding
    embedding technique (SoftEmbedding),    from https://arxiv.org/pdf/1708.00065.pdf
    In a nutshell, it keeps an embedding table for each continuous feature,
    which is represented as a weighted average of embeddings.

    Parameters
    ----------
    feature_config: Dict[str, FeatureConfig]
        This specifies what TableConfig to use for each feature. For shared embeddings, the same
        TableConfig can be used for multiple features.
    layer_norm: boolean
        When layer_norm is true, TabularLayerNorm will be used in post.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        feature_config: Dict[str, "FeatureConfig"],
        layer_norm: bool = True,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        **kwarg,
    ):
        if layer_norm:
            from transformers4rec.torch import TabularLayerNorm

            post = TabularLayerNorm.from_feature_config(feature_config)

        super().__init__(feature_config, pre=pre, post=post, aggregation=aggregation)

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        soft_embedding_cardinalities: Optional[Dict[str, int]] = None,
        soft_embedding_cardinality_default: int = 10,
        soft_embedding_dims: Optional[Dict[str, int]] = None,
        soft_embedding_dim_default: int = 8,
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
        layer_norm: bool = True,
        combiner: str = "mean",
        tags: Optional[Union[Tag, list, str]] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ) -> Optional["SoftEmbeddingFeatures"]:
        """
        Instantitates ``SoftEmbeddingFeatures`` from a ``DatasetSchema``.

        Parameters
        ----------
        schema : DatasetSchema
            Dataset schema
        soft_embedding_cardinalities : Optional[Dict[str, int]], optional
            The cardinality of the embedding table for each feature (key),
            by default None
        soft_embedding_cardinality_default : Optional[int], optional
            Default cardinality of the embedding table, when the feature
            is not found in ``soft_embedding_cardinalities``, by default 10
        soft_embedding_dims : Optional[Dict[str, int]], optional
            The dimension of the embedding table for each feature (key), by default None
        soft_embedding_dim_default : Optional[int], optional
            Default dimension of the embedding table, when the feature
            is not found in ``soft_embedding_dim_default``, by default 8
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]]
            Dict where keys are feature names and values are callable to initialize embedding tables
        combiner : Optional[str], optional
            Feature aggregation option, by default "mean"
        tags : Optional[Union[DefaultTags, list, str]], optional
            Tags to filter columns, by default None
        automatic_build : bool, optional
            Automatically infers input size from features, by default True
        max_sequence_length : Optional[int], optional
            Maximum sequence length for list features, by default None

        Returns
        -------
        Optional[SoftEmbeddingFeatures]
            Returns a ``SoftEmbeddingFeatures`` instance from the dataset schema
        """
        # TODO: propagate item-id from ITEM_ID tag

        if tags:
            schema = schema.select_by_tag(tags)

        soft_embedding_cardinalities = soft_embedding_cardinalities or {}
        soft_embedding_dims = soft_embedding_dims or {}
        embeddings_initializers = embeddings_initializers or {}

        sizes = {}
        cardinalities = schema.categorical_cardinalities()
        for col_name in schema.column_names:
            # If this is NOT a categorical feature
            if col_name not in cardinalities:
                embedding_size = soft_embedding_dims.get(col_name, soft_embedding_dim_default)
                cardinality = soft_embedding_cardinalities.get(
                    col_name, soft_embedding_cardinality_default
                )
                emb_initializer = embeddings_initializers.get(col_name, None)
                sizes[col_name] = (cardinality, embedding_size, emb_initializer)

        feature_config: Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim, emb_initializer) in sizes.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=name,
                    combiner=combiner,
                    initializer=emb_initializer,
                )
            )

        if not feature_config:
            return None

        output = cls(feature_config, layer_norm=layer_norm, **kwargs)

        if automatic_build and schema:
            output.build(
                get_output_sizes_from_schema(
                    schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                )
            )

        return output

    def table_to_embedding_module(self, table: "TableConfig") -> "SoftEmbedding":
        return SoftEmbedding(table.vocabulary_size, table.dim, table.initializer)


class TableConfig:
    def __init__(
        self,
        vocabulary_size: int,
        dim: int,
        initializer: Optional[Callable[[torch.Tensor], None]] = None,
        combiner: Text = "mean",
        name: Optional[Text] = None,
    ):
        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError("Invalid vocabulary_size {}.".format(vocabulary_size))

        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Invalid dim {}.".format(dim))

        if combiner not in ("mean", "sum", "sqrtn"):
            raise ValueError("Invalid combiner {}".format(combiner))

        if (initializer is not None) and (not callable(initializer)):
            raise ValueError("initializer must be callable if specified.")

        self.initializer: Callable[[torch.Tensor], None]
        if initializer is None:
            self.initializer = partial(torch.nn.init.normal_, mean=0.0, std=0.05)  # type: ignore
        else:
            self.initializer = initializer

        self.vocabulary_size = vocabulary_size
        self.dim = dim
        self.combiner = combiner
        self.name = name

    def __repr__(self):
        return (
            "TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, "
            "combiner={combiner!r}, name={name!r})".format(
                vocabulary_size=self.vocabulary_size,
                dim=self.dim,
                combiner=self.combiner,
                name=self.name,
            )
        )


class FeatureConfig:
    def __init__(
        self, table: TableConfig, max_sequence_length: int = 0, name: Optional[Text] = None
    ):
        self.table = table
        self.max_sequence_length = max_sequence_length
        self.name = name

    def __repr__(self):
        return (
            "FeatureConfig(table={table!r}, "
            "max_sequence_length={max_sequence_length!r}, name={name!r})".format(
                table=self.table, max_sequence_length=self.max_sequence_length, name=self.name
            )
        )


class SoftEmbedding(torch.nn.Module):
    """
    Soft-one hot encoding embedding technique, from https://arxiv.org/pdf/1708.00065.pdf
    In a nutshell, it represents a continuous feature as a weighted average of embeddings
    """

    def __init__(self, num_embeddings, embeddings_dim, emb_initializer=None):
        """

        Parameters
        ----------
        num_embeddings: Number of embeddings to use (cardinality of the embedding table).
        embeddings_dim: The dimension of the vector space for projecting the scalar value.
        embeddings_init_std: The standard deviation factor for normal initialization of the
            embedding matrix weights.
        emb_initializer: Dict where keys are feature names and values are callable to initialize
            embedding tables
        """

        assert (
            num_embeddings > 0
        ), "The number of embeddings for soft embeddings needs to be greater than 0"
        assert (
            embeddings_dim > 0
        ), "The embeddings dim for soft embeddings needs to be greater than 0"

        super(SoftEmbedding, self).__init__()
        self.embedding_table = torch.nn.Embedding(num_embeddings, embeddings_dim)
        if emb_initializer:
            emb_initializer(self.embedding_table.weight)

        self.projection_layer = torch.nn.Linear(1, num_embeddings, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        input_numeric = input_numeric.unsqueeze(-1)
        weights = self.softmax(self.projection_layer(input_numeric))
        soft_one_hot_embeddings = (weights.unsqueeze(-1) * self.embedding_table.weight).sum(-2)

        return soft_one_hot_embeddings
