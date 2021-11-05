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

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig, TableConfig

from merlin_standard_lib import Schema
from merlin_standard_lib.schema.tag import TagsType
from merlin_standard_lib.utils.doc_utils import docstring_parameter
from merlin_standard_lib.utils.embedding_utils import get_embedding_sizes_from_schema

from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    FilterFeatures,
    TabularAggregationType,
    TabularTransformationType,
)
from ..tabular.transformations import AsSparseFeatures
from ..typing import TabularData
from .base import InputBlock

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg


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
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
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
        name=None,
        add_default_pre=True,
        embedding_tables={},
        **kwargs,
    ):
        if not item_id and schema and schema.select_by_tag(["item_id"]).column_names:
            item_id = schema.select_by_tag(["item_id"]).column_names[0]

        if add_default_pre:
            embedding_pre = [FilterFeatures(list(feature_config.keys())), AsSparseFeatures()]
            pre = [embedding_pre, pre] if pre else embedding_pre  # type: ignore
        self.feature_config = feature_config
        self.item_id = item_id
        self.embedding_tables = embedding_tables

        super().__init__(
            pre=pre, post=post, aggregation=aggregation, name=name, schema=schema, **kwargs
        )

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        embedding_dims: Optional[Dict[str, int]] = None,
        embedding_dim_default: Optional[int] = 64,
        infer_embedding_sizes: bool = False,
        infer_embedding_sizes_multiplier: float = 2.0,
        embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
        combiner: Optional[str] = "mean",
        tags: Optional[TagsType] = None,
        item_id: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ) -> Optional["EmbeddingFeatures"]:
        schema_copy = schema.copy()

        if tags:
            schema_copy = schema_copy.select_by_tag(tags)

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

        output = cls(
            feature_config, item_id=item_id, schema=schema_copy, embedding_tables={}, **kwargs
        )

        return output

    def build(self, input_shapes):
        tables: Dict[str, TableConfig] = {}
        for name, feature in self.feature_config.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for name, table in tables.items():
            shape = (table.vocabulary_size, table.dim)
            if name not in self.embedding_tables:
                self.embedding_tables[name] = self.add_weight(
                    name="{}/embedding_weights".format(name),
                    trainable=True,
                    initializer=table.initializer,
                    shape=shape,
                )
        super().build(input_shapes)

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        embedded_outputs = {}
        for name, val in inputs.items():
            embedded_outputs[name] = self.lookup_feature(name, val)

        # Store raw item ids for masking and/or negative sampling
        # This makes this module stateful.
        if self.item_id:
            self.item_seq = self.item_ids(inputs)

        return embedded_outputs

    def compute_call_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        output_shapes = {}
        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape([batch_size, self.feature_config[name].table.dim])

        return output_shapes

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

        table: TableConfig = self.feature_config[name].table
        table_var = self.embedding_tables[table.name]
        if isinstance(val, tf.SparseTensor):
            out = tf.nn.safe_embedding_lookup_sparse(table_var, val, None, combiner=table.combiner)
        else:
            if output_sequence:
                out = tf.gather(table_var, tf.cast(val, tf.int32))
            else:
                if len(val.shape) > 1:
                    # TODO: Check if it is correct to retrieve only the 1st element
                    # of second dim for non-sequential multi-hot categ features
                    out = tf.gather(table_var, tf.cast(val, tf.int32)[:, 0])
                else:
                    out = tf.gather(table_var, tf.cast(val, tf.int32))

        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)

        return out

    def get_config(self):
        config = super().get_config()

        feature_configs = {}

        for key, val in self.feature_config.items():
            feature_config_dict = dict(name=val.name, max_sequence_length=val.max_sequence_length)

            feature_config_dict["table"] = serialize_table_config(val.table)
            feature_configs[key] = feature_config_dict

        config["feature_config"] = feature_configs
        if self.item_id:
            config["item_id"] = self.item_id
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize feature_config
        feature_configs, table_configs = {}, {}
        for key, val in config["feature_config"].items():
            feature_params = deepcopy(val)
            table_params = feature_params["table"]
            if "name" in table_configs:
                feature_params["table"] = table_configs["name"]
            else:
                table = deserialize_table_config(table_params)
                if table.name:
                    table_configs[table.name] = table
                feature_params["table"] = table
            feature_configs[key] = FeatureConfig(**feature_params)
        config["feature_config"] = feature_configs

        # Set `add_default_pre to False` since pre will be provided from the config
        config["add_default_pre"] = False

        return super().from_config(config)


def serialize_table_config(table_config: TableConfig) -> Dict[str, Any]:
    table = deepcopy(table_config.__dict__)
    if "initializer" in table:
        table["initializer"] = tf.keras.initializers.serialize(table["initializer"])
    if "optimizer" in table:
        table["optimizer"] = tf.keras.optimizers.serialize(table["optimizer"])

    return table


def deserialize_table_config(table_params: Dict[str, Any]) -> TableConfig:
    if "initializer" in table_params and table_params["initializer"]:
        table_params["initializer"] = tf.keras.initializers.deserialize(table_params["initializer"])
    if "optimizer" in table_params and table_params["optimizer"]:
        table_params["optimizer"] = tf.keras.optimizers.deserialize(table_params["optimizer"])
    table = TableConfig(**table_params)

    return table


def serialize_feature_config(feature_config: FeatureConfig) -> Dict[str, Any]:
    outputs = {}

    for key, val in feature_config.items():
        feature_config_dict = dict(name=val.name, max_sequence_length=val.max_sequence_length)
        feature_config_dict["table"] = serialize_table_config(feature_config_dict["table"])
        outputs[key] = feature_config_dict

    return outputs
