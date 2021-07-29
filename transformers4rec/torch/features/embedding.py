import math
from typing import Dict, Optional, Text, Union

import torch
import yaml

from transformers4rec.torch.utils.torch_utils import calculate_batch_size_from_input_size

from ...types import ColumnGroup, DefaultTags
from ..tabular import FilterFeatures, TabularModule


class TableConfig(object):
    def __init__(
        self,
        vocabulary_size: int,
        dim: int,
        # initializer: Optional[Callable[[Any], None]],
        # optimizer: Optional[_Optimizer] = None,
        combiner: Text = "mean",
        name: Optional[Text] = None,
    ):
        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError("Invalid vocabulary_size {}.".format(vocabulary_size))

        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Invalid dim {}.".format(dim))

        if combiner not in ("mean", "sum", "sqrtn"):
            raise ValueError("Invalid combiner {}".format(combiner))

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


class FeatureConfig(object):
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
    Soft-one hot encoding embedding, from https://arxiv.org/pdf/1708.00065.pdf
    """

    def __init__(self, num_embeddings, embeddings_dim, embeddings_init_std=0.05):
        """

        Parameters
        ----------
        num_embeddings: Number of embeddings to use.
        embeddings_dim: The dimension of the vector space for projecting the scalar value.
        embeddings_init_std: The standard deviation factor for normal initialization of the
            embedding matrix weights.
        """
        super(SoftEmbedding, self).__init__()
        self.embedding_table = torch.nn.Embedding(num_embeddings, embeddings_dim)
        with torch.no_grad():
            self.embedding_table.weight.normal_(0.0, embeddings_init_std)
        self.projection_layer = torch.nn.Linear(1, num_embeddings, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        weights = self.softmax(self.projection_layer(input_numeric))
        soft_one_hot_embeddings = (weights.unsqueeze(-1) * self.embedding_table.weight).sum(-2)

        return soft_one_hot_embeddings


class EmbeddingFeatures(TabularModule):
    def __init__(self, feature_config: Dict[str, FeatureConfig], item_id=None, **kwargs):
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

    def table_to_embedding_module(self, table: TableConfig) -> torch.nn.Module:
        return torch.nn.EmbeddingBag(table.vocabulary_size, table.dim, mode=table.combiner)

    @classmethod
    def from_column_group(
        cls,
        column_group: ColumnGroup,
        embedding_dims=None,
        default_embedding_dim=64,
        infer_embedding_sizes=False,
        combiner="mean",
        tags=None,
        tags_to_filter=None,
        **kwargs
    ) -> Optional["EmbeddingFeatures"]:
        # TODO: propogate item-id from ITEM_ID tag
        if tags:
            column_group = column_group.get_tagged(tags, tags_to_filter=tags_to_filter)

        if infer_embedding_sizes:
            sizes = column_group.embedding_sizes()
        else:
            if not embedding_dims:
                embedding_dims = {}
            sizes = {}
            cardinalities = column_group.cardinalities()
            for key, cardinality in cardinalities.items():
                embedding_size = embedding_dims.get(key, default_embedding_dim)
                sizes[key] = (cardinality, embedding_size)

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

        return cls(feature_config, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, dict], str],
        embedding_dims=None,
        default_embedding_dim=64,
        infer_embedding_sizes=False,
        combiner="mean",
        tags=None,
        tags_to_filter=None,
        **kwargs
    ) -> Optional["EmbeddingFeatures"]:
        if isinstance(config, str):
            # load config from specified path
            with open(config) as yaml_file:
                config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not isinstance(tags, list):
            tags = [tags]

        column_names_to_filter = []
        if tags_to_filter:
            column_names_to_filter = [
                key for key, val in config.items() if all(x in val["tags"] for x in tags_to_filter)
            ]
        if tags:
            output_cols = [
                {key: val} for key, val in config.items() if all(x in val["tags"] for x in tags)
            ]
        output_cols = {
            k: v for d in output_cols for k, v in d.items() if k not in column_names_to_filter
        }

        if infer_embedding_sizes:
            sizes = [
                {
                    key: (
                        val["cardinality"],
                        cls.get_embedding_size_from_cardinality(val["cardinality"]),
                    )
                }
                for key, val in output_cols.items()
            ]
        else:
            if not embedding_dims:
                embedding_dims = {}
            sizes = {}
            cardinalities = {key: val["cardinality"] for key, val in output_cols.items()}
            for key, cardinality in cardinalities.items():
                embedding_size = embedding_dims.get(key, default_embedding_dim)
                sizes[key] = (cardinality, embedding_size)

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

        return cls(feature_config, **kwargs)

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

        return embedded_outputs

    def forward_output_size(self, input_sizes):
        sizes = {}
        batch_size = calculate_batch_size_from_input_size(input_sizes)
        for name, feature in self.feature_config.items():
            sizes[name] = torch.Size([batch_size, feature.table.dim])

        return super().forward_output_size(sizes)

    @staticmethod
    def get_embedding_size_from_cardinality(cardinality, multiplier=2.0):
        # A rule-of-thumb from Google.
        embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
        return embedding_size


class SoftEmbeddingFeatures(EmbeddingFeatures):
    def table_to_embedding_module(self, table: TableConfig) -> SoftEmbedding:
        return SoftEmbedding(table.vocabulary_size, table.dim)
