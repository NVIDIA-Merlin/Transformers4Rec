import math
from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig, TableConfig

from ...types import DatasetSchema
from ..tabular import AsSparseFeatures, FilterFeatures
from .base import InputLayer

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg


class EmbeddingFeatures(InputLayer):
    def __init__(self, feature_config: Dict[str, FeatureConfig], item_id=None, **kwargs):
        self.filter_features = FilterFeatures(list(feature_config.keys()))
        self.convert_to_sparse = AsSparseFeatures()
        self.embeddings = feature_config
        self.item_id = item_id
        super().__init__(**kwargs)

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        embedding_dims=None,
        default_embedding_dim=64,
        infer_embedding_sizes=False,
        combiner="mean",
        tags=None,
        item_id: Optional[str] = None,
        **kwargs
    ) -> Optional["EmbeddingFeatures"]:
        if tags:
            schema = schema.select_by_tag(tags)

        if not item_id and schema.select_by_tag(["item_id"]).column_names:
            item_id = schema.select_by_tag(["item_id"]).column_names[0]

        if infer_embedding_sizes:
            sizes = schema.embedding_sizes()
        else:
            if not embedding_dims:
                embedding_dims = {}
            sizes = {}
            cardinalities = schema.cardinalities()
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
                    initializer=init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(dim)),
                )
            )

        if not feature_config:
            return None

        return cls(feature_config, item_id=item_id, **kwargs)

    def build(self, input_shapes):
        self.embedding_tables = {}
        tables: Dict[str, TableConfig] = {}
        for name, feature in self.embeddings.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for name, table in tables.items():
            shape = (table.vocabulary_size, table.dim)
            self.embedding_tables[name] = self.add_weight(
                name="{}/embedding_weights".format(name),
                trainable=True,
                initializer=table.initializer,
                shape=shape,
            )
        super().build(input_shapes)

    @property
    def item_embedding_table(self):
        assert self.item_id is not None

        return self.embedding_tables[self.item_id]

    def lookup_feature(self, name, val):
        table: TableConfig = self.embeddings[name].table
        table_var = self.embedding_tables[table.name]
        if isinstance(val, tf.SparseTensor):
            return tf.nn.safe_embedding_lookup_sparse(
                table_var, tf.cast(val, tf.int32), None, combiner=table.combiner
            )

        # embedded_outputs[name] = tf.gather(table_var, tf.cast(val, tf.int32))
        return tf.gather(table_var, tf.cast(val, tf.int32)[:, 0])

    def compute_output_shape(self, input_shapes):
        input_shapes = self.filter_features.compute_output_shape(input_shapes)
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        output_shapes = {}

        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape([batch_size, self.embeddings[name].table.dim])

        return super().compute_output_shape(output_shapes)

    def call(self, inputs, **kwargs):
        embedded_outputs = {}
        sparse_inputs = self.convert_to_sparse(self.filter_features(inputs))
        for name, val in sparse_inputs.items():
            embedded_outputs[name] = self.lookup_feature(name, val)

        return embedded_outputs

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]
