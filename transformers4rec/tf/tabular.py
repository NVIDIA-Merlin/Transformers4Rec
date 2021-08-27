from functools import reduce
from typing import Optional

import tensorflow as tf

from ..types import DatasetSchema
from . import aggregation as agg
from .utils.tf_utils import calculate_batch_size_from_input_shapes


class FilterFeatures(tf.keras.layers.Layer):
    def __init__(
        self, to_include, trainable=False, name=None, dtype=None, dynamic=False, pop=False, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.to_include = to_include
        self.pop = pop

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.to_include}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def compute_output_shape(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.to_include}

    def get_config(self):
        return {
            "to_include": self.to_include,
        }


class AsTabular(tf.keras.layers.Layer):
    def __init__(
        self, output_name, trainable=False, name=None, dtype=None, dynamic=False, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.output_name = output_name

    def call(self, inputs, **kwargs):
        return {self.output_name: inputs}

    def get_config(self):
        return {
            "axis": self.axis,
        }


class TabularLayer(tf.keras.layers.Layer):
    def __init__(
        self, aggregation=None, trainable=True, name=None, dtype=None, dynamic=False, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.set_aggregation(aggregation)

    @property
    def aggregation(self):
        return self._aggregation

    def set_aggregation(self, value):
        if value:
            self._aggregation = agg.aggregation_registry.parse(value)
        else:
            self._aggregation = None

    def __call__(
        self,
        inputs,
        pre=None,
        post=None,
        merge_with=None,
        stack_outputs=False,
        concat_outputs=False,
        aggregation=None,
        filter_columns=None,
        training=False,
        **kwargs
    ):
        post_op = getattr(self, "aggregation", None)
        if concat_outputs:
            post_op = agg.ConcatFeatures()
        if stack_outputs:
            post_op = agg.StackFeatures()
        if aggregation:
            post_op = agg.aggregation_registry.parse(aggregation)
        if filter_columns:
            pre = FilterFeatures(filter_columns)
        if pre:
            inputs = pre(inputs)
        outputs = super().__call__(inputs, training=training, **kwargs)

        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = (
                    layer_or_tensor(inputs, training=training, **kwargs)
                    if callable(layer_or_tensor)
                    else layer_or_tensor
                )
                outputs.update(to_add)

        if isinstance(outputs, dict) and post_op:
            outputs = post_op(outputs)

        return outputs

    def compute_output_shape(self, input_shapes):
        if self.aggregation:
            return self.aggregation.compute_output_shape(input_shapes)

        return input_shapes

    def apply_to_all(self, inputs, columns_to_filter=None):
        if columns_to_filter:
            inputs = FilterFeatures(columns_to_filter)(inputs)
        outputs = tf.nest.map_structure(self, inputs)

        return outputs

    def repr_ignore(self):
        return []

    def repr_extra(self):
        return []

    def repr_add(self):
        return []

    @staticmethod
    def calculate_batch_size_from_input_shapes(input_shapes):
        return calculate_batch_size_from_input_shapes(input_shapes)

    @classmethod
    def from_schema(cls, schema: DatasetSchema, tags=None, **kwargs) -> Optional["TabularLayer"]:
        if tags:
            schema = schema.select_by_tag(tags)

        if not schema.columns:
            return None

        return cls.from_features(schema.column_names, **kwargs)

    @classmethod
    def from_features(cls, features, **kwargs):
        return features >> cls(**kwargs)

    def __rrshift__(self, other):
        from .block.base import right_shift_layer

        return right_shift_layer(self, other)


class MergeTabular(TabularLayer):
    def __init__(
        self,
        *to_merge,
        aggregation=None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(aggregation, trainable, name, dtype, dynamic, **kwargs)
        if all(isinstance(x, dict) for x in to_merge):
            to_merge = reduce(lambda a, b: dict(a, **b), to_merge)
            self.to_merge = to_merge
        else:
            self.to_merge = list(to_merge)

    @property
    def merge_values(self):
        if isinstance(self.to_merge, dict):
            return list(self.to_merge.values())

        return self.to_merge

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        for layer in self.merge_values:
            outputs.update(layer(inputs))

        return outputs

    def compute_output_shape(self, input_shape):
        output_shapes = {}

        for layer in self.merge_values:
            output_shapes.update(layer.compute_output_shape(input_shape))

        return super(MergeTabular, self).compute_output_shape(output_shapes)

    def get_config(self):
        return {"merge_layers": tf.keras.utils.serialize_keras_object(self.merge_layers)}


def merge_tabular(self, other, aggregation=None, **kwargs):
    return MergeTabular(self, other, aggregation=aggregation, **kwargs)


TabularLayer.__add__ = merge_tabular
TabularLayer.merge = merge_tabular


class AsSparseFeatures(TabularLayer):
    def call(self, inputs, **kwargs):
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.columns}


class AsDenseFeatures(TabularLayer):
    def call(self, inputs, **kwargs):
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_tensor()
            else:
                outputs[name] = val

        return outputs
