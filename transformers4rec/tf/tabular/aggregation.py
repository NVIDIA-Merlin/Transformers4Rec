from functools import reduce
from typing import List

import tensorflow as tf

from ...utils.schema import requires_schema
from ..typing import TabularData
from ..utils.tf_utils import calculate_batch_size_from_input_shapes
from .tabular import TabularAggregation, tabular_aggregation_registry

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg


@tabular_aggregation_registry.register("concat")
class ConcatFeatures(TabularAggregation):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        return tf.concat(
            tf.nest.flatten(tf.nest.map_structure(self.flatten, inputs)), axis=self.axis
        )

    def compute_output_shape(self, input_shapes):
        batch_size = calculate_batch_size_from_input_shapes(input_shapes)

        return batch_size, sum([i[1] for i in input_shapes.values()])

    def repr_ignore(self) -> List[str]:
        return ["flatten"]

    def get_config(self):
        return {
            "axis": self.axis,
        }


@tabular_aggregation_registry.register_with_multiple_names("sequential_concat", "sequential-concat")
class SequentialConcatFeatures(TabularAggregation):
    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        tensors = []
        for name in sorted(inputs.keys()):
            val = inputs[name]
            if len(val.shape) == 2:
                val = tf.expand_dims(val, axis=-1)
            tensors.append(val)

        return tf.concat(tensors, axis=-1)

    def compute_output_shape(self, input_size):
        batch_size = calculate_batch_size_from_input_shapes(input_size)
        converted_input_size = {}
        for key, val in input_size.items():
            if len(val) == 2:
                converted_input_size[key] = val + (1,)
            else:
                converted_input_size[key] = val

        return (
            batch_size,
            list(input_size.values())[0][1],
            sum([i[-1] for i in converted_input_size.values()]),
        )


@tabular_aggregation_registry.register("stack")
class StackFeatures(TabularAggregation):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        return tf.stack(
            tf.nest.flatten(tf.nest.map_structure(self.flatten, inputs)), axis=self.axis
        )

    def compute_output_shape(self, input_shapes):
        batch_size = calculate_batch_size_from_input_shapes(input_shapes)
        last_dim = list(input_shapes.values())[0][-1]

        return batch_size, len(input_shapes), last_dim

    def repr_ignore(self) -> List[str]:
        return ["flatten"]

    def get_config(self):
        return {
            "axis": self.axis,
        }


class ElementwiseFeatureAggregation(TabularAggregation):
    def _check_input_shapes_equal(self, inputs):
        all_input_shapes_equal = reduce((lambda a, b: a.shape == b.shape), inputs.values())
        if not all_input_shapes_equal:
            raise ValueError(
                "The shapes of all input features are not equal, which is required for element-wise"
                " aggregation: {}".format({k: v.shape for k, v in inputs.items()})
            )


@tabular_aggregation_registry.register("element-wise-sum")
class ElementwiseSum(ElementwiseFeatureAggregation):
    def __init__(self):
        super().__init__()
        self.stack = StackFeatures(axis=0)

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._check_input_shapes_equal(inputs)
        return tf.reduce_sum(self.stack(inputs), axis=0)

    def compute_output_shape(self, input_shape):
        batch_size = calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim


@tabular_aggregation_registry.register("element-wise-sum-item-multi")
@requires_schema
class ElementwiseSumItemMulti(ElementwiseFeatureAggregation):
    def __init__(self, schema=None):
        super().__init__()
        self.stack = StackFeatures(axis=0)
        if schema:
            self.set_schema(schema)
        self.item_id_col_name = None

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        item_id_inputs = self.schema.get_item_ids_from_inputs(inputs)
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
