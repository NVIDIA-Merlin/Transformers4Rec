from typing import List

import tensorflow as tf

from ..utils.registry import Registry
from .typing import TabularData
from transformers4rec.tf.utils.tf_utils import calculate_batch_size_from_input_shapes

aggregators = Registry.class_registry("tf.aggregators")


class FeatureAggregator(tf.keras.layers.Layer):
    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        return super(FeatureAggregator, self).call(inputs, **kwargs)


@aggregators.register("concat")
class ConcatFeatures(FeatureAggregator):
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


@aggregators.register("stack")
class StackFeatures(FeatureAggregator):
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
        last_dim = [i for i in input_shapes.values()][0][-1]

        return batch_size, len(input_shapes), last_dim

    def repr_ignore(self) -> List[str]:
        return ["flatten"]

    def get_config(self):
        return {
            "axis": self.axis,
        }
