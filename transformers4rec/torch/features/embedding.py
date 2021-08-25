from functools import partial
from typing import Any, Callable, Dict, Optional, Text, Union

import torch

from transformers4rec.torch.utils.torch_utils import (
    calculate_batch_size_from_input_size,
    get_output_sizes_from_schema,
)

from ...types import DatasetSchema, DefaultTags
from ..tabular import FilterFeatures, TabularModule


class EmbeddingFeatures(TabularModule):
    def __init__(
        self,
        feature_config: Dict[str, "FeatureConfig"],
        item_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.item_id = item_id
        self.feature_config = feature_config
        self.filter_features = FilterFeatures(list(feature_config.keys()))

        embedding_tables = {}
        tables: Dict[str, TableConfig] = {}
        for name, feature in self.feature_config.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for name, table in tables.items():
            embedding_tables[name] = self.table_to_embedding_module(table)

        self.embedding_tables = torch.nn.ModuleDict(embedding_tables)

    @property
    def item_embedding_table(self):
        assert self.item_id is not None

        return self.embedding_tables[self.item_id]

    def table_to_embedding_module(self, table: "TableConfig") -> torch.nn.Module:
        embedding_table = torch.nn.EmbeddingBag(
            table.vocabulary_size, table.dim, mode=table.combiner
        )

        if table.initializer is not None:
            table.initializer(embedding_table.weight)
        return embedding_table

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        embedding_dims: Optional[Dict[str, int]] = None,
        default_embedding_dim: Optional[int] = 64,
        infer_embedding_sizes: bool = False,
        infer_embedding_sizes_multiplier: Optional[float] = 2.0,
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
        combiner: Optional[str] = "mean",
        tags: Optional[Union[DefaultTags, list, str]] = None,
        item_id: Optional[str] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
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
        # TODO: propogate item-id from ITEM_ID tag

        if tags:
            schema = schema.select_by_tag(tags)

        if not item_id and schema.select_by_tag(["item_id"]).column_names:
            item_id = schema.select_by_tag(["item_id"]).column_names[0]

        if infer_embedding_sizes:
            embedding_dims = schema.embedding_sizes_v2(infer_embedding_sizes_multiplier)

        embedding_dims = embedding_dims or {}
        embeddings_initializers = embeddings_initializers or {}

        emb_config = {}
        cardinalities = schema.cardinalities()
        for key, cardinality in cardinalities.items():
            embedding_size = embedding_dims.get(key, default_embedding_dim)
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

        output = cls(feature_config, item_id=item_id, **kwargs)

        if automatic_build and schema._schema:
            output.build(
                get_output_sizes_from_schema(
                    schema._schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                )
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
                if len(val.shape) <= 1:
                    val = val.unsqueeze(0)
                embedded_outputs[name] = self.embedding_tables[name](val)
        # Store raw item ids for masking and/or negative sampling
        # This makes this module stateful.
        if self.item_id:
            self.item_seq = self.item_ids(inputs)

        return embedded_outputs

    def forward_output_size(self, input_sizes):
        sizes = {}
        batch_size = calculate_batch_size_from_input_size(input_sizes)
        for name, feature in self.feature_config.items():
            sizes[name] = torch.Size([batch_size, feature.table.dim])

        return super().forward_output_size(sizes)


class SoftEmbeddingFeatures(EmbeddingFeatures):
    """
    Encapsulate continuous features encoded using the Soft-one hot encoding
    embedding technique (SoftEmbedding),    from https://arxiv.org/pdf/1708.00065.pdf
    In a nutshell, it keeps an embedding table for each continuous feature,
    which is represented as a weighted average of embeddings.
    """

    def __init__(
        self,
        feature_config: Dict[str, "FeatureConfig"],
        soft_embeddings_init_std: float = 0.05,
        **kwargs,
    ):
        self.soft_embeddings_init_std = soft_embeddings_init_std
        super().__init__(feature_config, **kwargs)

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        soft_embedding_cardinalities: Optional[Dict[str, int]] = None,
        default_soft_embedding_cardinality: Optional[int] = 10,
        soft_embedding_dims: Optional[Dict[str, int]] = None,
        default_soft_embedding_dim: Optional[int] = 8,
        combiner: Optional[str] = "mean",
        tags: Optional[Union[DefaultTags, list, str]] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
        soft_embeddings_init_std=0.05,
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
        default_soft_embedding_cardinality : Optional[int], optional
            Default cardinality of the embedding table, when the feature
            is not found in ``soft_embedding_cardinalities``, by default 10
        soft_embedding_dims : Optional[Dict[str, int]], optional
            The dimension of the embedding table for each feature (key), by default None
        default_soft_embedding_dim : Optional[int], optional
            Default dimension of the embedding table, when the feature
            is not found in ``default_soft_embedding_dim``, by default 8
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
        # TODO: propogate item-id from ITEM_ID tag

        if tags:
            schema = schema.select_by_tag(tags)

        soft_embedding_cardinalities = soft_embedding_cardinalities or {}
        soft_embedding_dims = soft_embedding_dims or {}

        sizes = {}
        cardinalities = schema.cardinalities()
        for col_name in schema.column_names:
            # If this is NOT a categorical feature
            if col_name not in cardinalities:
                embedding_size = soft_embedding_dims.get(col_name, default_soft_embedding_dim)
                cardinality = soft_embedding_cardinalities.get(
                    col_name, default_soft_embedding_cardinality
                )
                sizes[col_name] = (cardinality, embedding_size)

        feature_config: Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim) in sizes.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=name,
                    combiner=combiner,
                )
            )

        if not feature_config:
            return None

        output = cls(feature_config, soft_embeddings_init_std, **kwargs)

        if automatic_build and schema._schema:
            output.build(
                get_output_sizes_from_schema(
                    schema._schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                )
            )

        return output

    def table_to_embedding_module(self, table: "TableConfig") -> "SoftEmbedding":
        return SoftEmbedding(table.vocabulary_size, table.dim, self.soft_embeddings_init_std)


class TableConfig:
    def __init__(
        self,
        vocabulary_size: int,
        dim: int,
        initializer: Optional[Callable[[Any], None]] = None,
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
        if initializer is None:
            initializer = partial(torch.nn.init.normal_, mean=0.0, std=0.05)

        self.vocabulary_size = vocabulary_size
        self.dim = dim
        self.combiner = combiner
        self.name = name
        self.initializer = initializer

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

    def __init__(self, num_embeddings, embeddings_dim, embeddings_init_std=0.05):
        """

        Parameters
        ----------
        num_embeddings: Number of embeddings to use (cardinality of the embedding table).
        embeddings_dim: The dimension of the vector space for projecting the scalar value.
        embeddings_init_std: The standard deviation factor for normal initialization of the
            embedding matrix weights.
        """

        assert (
            num_embeddings > 0
        ), "The number of embeddings for soft embeddings needs to be greater than 0"
        assert (
            embeddings_dim > 0
        ), "The embeddings dim for soft embeddings needs to be greater than 0"

        super(SoftEmbedding, self).__init__()
        self.embedding_table = torch.nn.Embedding(num_embeddings, embeddings_dim)
        with torch.no_grad():
            self.embedding_table.weight.normal_(0.0, embeddings_init_std)
        self.projection_layer = torch.nn.Linear(1, num_embeddings, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        input_numeric = input_numeric.unsqueeze(-1)
        weights = self.softmax(self.projection_layer(input_numeric))
        soft_one_hot_embeddings = (weights.unsqueeze(-1) * self.embedding_table.weight).sum(-2)

        return soft_one_hot_embeddings
