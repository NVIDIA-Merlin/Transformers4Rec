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

import tensorflow as tf

from ...config.schema import requires_schema
from ..typing import TabularData
from ..utils.tf_utils import calculate_batch_size_from_input_shapes
from .tabular import TabularAggregation, tabular_aggregation_registry

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg


@tabular_aggregation_registry.register("concat")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class ConcatFeatures(TabularAggregation):
    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._maybe_expand_non_sequential_features(inputs)
        self._check_concat_shapes(inputs)

        tensors = []
        for name in sorted(inputs.keys()):
            val = inputs[name]
            tensors.append(val)

        return tf.concat(tensors, axis=-1)

    def compute_output_shape(self, input_shapes):
        agg_dim = sum([i[-1] for i in input_shapes.values()])
        output_size = self._get_agg_output_size(input_shapes, agg_dim)
        return output_size


@tabular_aggregation_registry.register("stack")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class StackFeatures(TabularAggregation):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._maybe_expand_non_sequential_features(inputs)
        self._check_concat_shapes(inputs)

        tensors = []
        for name in sorted(inputs.keys()):
            val = inputs[name]
            tensors.append(val)

        return tf.stack(tensors, axis=self.axis)

    def compute_output_shape(self, input_shapes):
        agg_dim = list(input_shapes.values())[0][-1]
        output_size = self._get_agg_output_size(input_shapes, agg_dim)
        return output_size

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis

        return config


class ElementwiseFeatureAggregation(TabularAggregation):
    def _check_input_shapes_equal(self, inputs):
        all_input_shapes_equal = len(set([tuple(x.shape) for x in inputs.values()])) == 1
        if not all_input_shapes_equal:
            raise ValueError(
                "The shapes of all input features are not equal, which is required for element-wise"
                " aggregation: {}".format({k: v.shape for k, v in inputs.items()})
            )


@tabular_aggregation_registry.register("element-wise-sum")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class ElementwiseSum(ElementwiseFeatureAggregation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stack = StackFeatures(axis=0)

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._maybe_expand_non_sequential_features(inputs)
        self._check_input_shapes_equal(inputs)

        return tf.reduce_sum(self.stack(inputs), axis=0)

    def compute_output_shape(self, input_shape):
        batch_size = calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim


@tabular_aggregation_registry.register("element-wise-sum-item-multi")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
@requires_schema
class ElementwiseSumItemMulti(ElementwiseFeatureAggregation):
    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)
        self.stack = StackFeatures(axis=0)
        if schema:
            self.set_schema(schema)
        self.item_id_col_name = None

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        item_id_inputs = self.get_item_ids_from_inputs(inputs)
        self._maybe_expand_non_sequential_features(inputs)
        self._check_input_shapes_equal(inputs)

        other_inputs = {k: v for k, v in inputs.items() if k != self.schema.item_id_column_name}
        # Sum other inputs when there are multiple features.
        if len(other_inputs) > 1:
            other_inputs = tf.reduce_sum(self.stack(other_inputs), axis=0)
        else:
            other_inputs = list(other_inputs.values())[0]
        result = item_id_inputs * other_inputs
        return result

    def compute_output_shape(self, input_shape):
        batch_size = calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim

    def get_config(self):
        config = super().get_config()
        if self.schema:
            config["schema"] = self.schema.to_json()

        return config
