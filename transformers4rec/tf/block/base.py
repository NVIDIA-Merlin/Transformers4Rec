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

import copy
import sys
from typing import Union

import six
import tensorflow as tf

from merlin_standard_lib.utils.misc_utils import filter_kwargs
from transformers4rec.config.schema import SchemaMixin


class Block(SchemaMixin, tf.keras.layers.Layer):
    def to_model(self, prediction_task_or_head, inputs=None, **kwargs):
        from ..model.base import Head, Model, PredictionTask

        if isinstance(prediction_task_or_head, PredictionTask):
            head = prediction_task_or_head.to_head(self, inputs=inputs, **kwargs)
        elif isinstance(prediction_task_or_head, Head):
            head = prediction_task_or_head
        else:
            raise ValueError(
                "`prediction_task_or_head` needs to be a `Head` or `PredictionTask` "
                f"found: {type(prediction_task_or_head)}"
            )

        return Model(head, **kwargs)

    def as_tabular(self, name=None):
        from ..tabular.base import AsTabular

        if not name:
            name = self.name

        return SequentialBlock([self, AsTabular(name)])


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class SequentialBlock(Block):
    """The SequentialLayer represents a sequence of Keras layers.
    It is a Keras Layer that can be used instead of tf.keras.layers.Sequential,
    which is actually a Keras Model.  In contrast to keras Sequential, this
    layer can be used as a pure Layer in tf.functions and when exporting
    SavedModels, without having to pre-declare input and output shapes.  In turn,
    this layer is usable as a preprocessing layer for TF Agents Networks, and
    can be exported via PolicySaver.
    Usage::

        c = SequentialLayer([layer1, layer2, layer3])
        output = c(inputs)    # Equivalent to: output = layer3(layer2(layer1(inputs)))
    """

    def __init__(self, layers, filter_features=None, block_name=None, **kwargs):
        """Create a composition.

        Parameters
        ----------
        layers:
            A list or tuple of layers to compose.
        **kwargs:
            Arguments to pass to `Keras` layer initializer, including `name`.

        Raises
        ------
        TypeError:
            If any of the layers are not instances of keras `Layer`.
        """
        self.block_name = block_name
        for layer in layers:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise TypeError(
                    "Expected all layers to be instances of keras Layer, but saw: '{}'".format(
                        layer
                    )
                )

        super(SequentialBlock, self).__init__(**kwargs)
        if filter_features:
            from ..tabular.base import FilterFeatures

            self.layers = [FilterFeatures(filter_features), *copy.copy(layers)]
        else:
            self.layers = copy.copy(layers)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.compute_output_shape(output_shape)
        return output_shape

    def compute_output_signature(self, input_signature):
        output_signature = input_signature
        for layer in self.layers:
            output_signature = layer.compute_output_signature(output_signature)
        return output_signature

    def build(self, input_shape=None):
        from ..tabular.base import TabularBlock

        last_layer = None
        for layer in self.layers:
            try:
                layer.build(input_shape)
            except TypeError:
                t, v, tb = sys.exc_info()
                if isinstance(input_shape, dict) and isinstance(last_layer, TabularBlock):
                    v = TypeError(
                        f"Couldn't build {layer}, "
                        f"did you forget to add aggregation to {last_layer}?"
                    )
                six.reraise(t, v, tb)
            input_shape = layer.compute_output_shape(input_shape)
            last_layer = layer
        self.built = True

    def set_schema(self, schema=None):
        for layer in self.layers:
            self._maybe_set_schema(layer, schema)

        return super().set_schema(schema)

    def _get_name(self):
        return self.block_name if self.block_name else f"{self.__class__.__name__}"

    @property
    def inputs(self):
        from transformers4rec.tf import TabularFeatures, TabularSequenceFeatures

        first = list(self)[0]
        if isinstance(first, (TabularSequenceFeatures, TabularFeatures)):
            return first

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = {}
        for layer in self.layers:
            for v in layer.trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def non_trainable_weights(self):
        weights = {}
        for layer in self.layers:
            for v in layer.non_trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def trainable(self):
        return all(layer.trainable for layer in self.layers)

    @trainable.setter
    def trainable(self, value):
        for layer in self.layers:
            layer.trainable = value

    @property
    def losses(self):
        values = set()
        for layer in self.layers:
            values.update(layer.losses)
        return list(values)

    @property
    def regularizers(self):
        values = set()
        for layer in self.layers:
            values.update(layer.regularizers)
        return list(values)

    def call(self, inputs, **kwargs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                filtered_kwargs = filter_kwargs(kwargs, layer, filter_positional_or_keyword=False)
                outputs = layer(outputs, **filtered_kwargs)
            else:
                outputs = layer(outputs)

        return outputs

    def get_config(self):
        config = {}
        for i, layer in enumerate(self.layers):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

        return config

    def __getitem__(self, key):
        return self.layers[key]

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        return SequentialBlock(layers)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def __rshift__(self, other):
        # pylint: disable=arguments-out-of-order
        return right_shift_layer(other, self)


BlockType = Union[tf.keras.layers.Layer, Block]


def right_shift_layer(self, other):
    from ..tabular.base import FilterFeatures

    if isinstance(other, list):
        left_side = [FilterFeatures(other)]
    else:
        left_side = other.layers if isinstance(other, SequentialBlock) else [other]
    right_side = self.layers if isinstance(self, SequentialBlock) else [self]

    return SequentialBlock(left_side + right_side)
