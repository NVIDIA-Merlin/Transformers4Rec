from typing import Any, Callable, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig, TableConfig

from transformers4rec.utils.tags import DefaultTags

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
        embedding_dims: Optional[Dict[str, int]] = None,
        embedding_dim_default: Optional[int] = 64,
        infer_embedding_sizes: bool = False,
        infer_embedding_sizes_multiplier: Optional[float] = 2.0,
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
        combiner: Optional[str] = "mean",
        tags: Optional[Union[DefaultTags, list, str]] = None,
        item_id: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ) -> Optional["EmbeddingFeatures"]:
        if tags:
            schema = schema.select_by_tag(tags)

        if not item_id and schema.select_by_tag(["item_id"]).column_names:
            item_id = schema.select_by_tag(["item_id"]).column_names[0]

        if infer_embedding_sizes:
            embedding_dims = schema.embedding_sizes(infer_embedding_sizes_multiplier)

        embedding_dims = embedding_dims or {}
        embeddings_initializers = embeddings_initializers or {}

        emb_config = {}
        cardinalities = schema.cardinalities()
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

        output = cls(feature_config, item_id=item_id, **kwargs)

        return output

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

    def item_ids(self, inputs) -> tf.Tensor:
        return inputs[self.item_id]

    def lookup_feature(self, name, val, output_sequence=False):
        dtype = backend.dtype(val)
        if dtype != "int32" and dtype != "int64":
            val = tf.cast(val, "int32")

        table: TableConfig = self.embeddings[name].table
        table_var = self.embedding_tables[table.name]
        if isinstance(val, tf.SparseTensor):
            out = tf.nn.safe_embedding_lookup_sparse(table_var, val, None, combiner=table.combiner)
        else:
            if output_sequence:
                out = tf.gather(table_var, tf.cast(val, tf.int32))
            else:
                out = tf.gather(table_var, tf.cast(val, tf.int32)[:, 0])

        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)

        return out

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

        # sparse_inputs = self.convert_to_sparse(self.filter_features(inputs))
        for name, val in sparse_inputs.items():
            embedded_outputs[name] = self.lookup_feature(name, val)

        # Store raw item ids for masking and/or negative sampling
        # This makes this module stateful.
        if self.item_id:
            self.item_seq = self.item_ids(inputs)

        return embedded_outputs

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]
