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
import abc
from typing import Dict, Union

import tensorflow as tf

from ..typing import TabularData


class LossMixin(abc.ABC):
    """Mixin to use for Keras Layers that can calculate a loss."""

    def compute_loss(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        compute_metrics=True,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Compute the loss on a batch of data.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            TODO
        targets: Union[torch.Tensor, TabularData]
            TODO
        training: bool, default=False
        """
        raise NotImplementedError()


class MetricsMixin(abc.ABC):
    """Mixin to use for Keras Layers that can calculate metrics."""

    def calculate_metrics(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        mode: str = "val",
        forward=True,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        """Calculate metrics on a batch of data, each metric is stateful and this updates the state.

        The state of each metric can be retrieved by calling the `metric_results` method.

        Parameters
        ----------
        inputs: Union[tf.Tensor, TabularData]
            TODO
        targets: Union[tf.Tensor, TabularData]
            TODO
        forward: bool, default True

        mode: str, default="val"

        """
        raise NotImplementedError()

    def metric_results(self, mode: str = None) -> Dict[str, Union[float, tf.Tensor]]:
        """Returns the current state of each metric.

        The state is typically updated each batch by calling the `calculate_metrics` method.

        Parameters
        ----------
        mode: str, default="val"

        Returns
        -------
        Dict[str, Union[float, tf.Tensor]]
        """
        raise NotImplementedError()

    def reset_metrics(self):
        """Reset all metrics."""
        raise NotImplementedError()


def get_output_sizes_from_schema(schema, batch_size=0, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
        if feature.HasField("value_count"):
            sizes[name] = tf.TensorShape(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        elif feature.HasField("shape"):
            sizes[name] = tf.TensorShape([batch_size] + [d.size for d in feature.shape.dim])
        else:
            sizes[name] = tf.TensorShape([batch_size, 1])

    return sizes


def calculate_batch_size_from_input_shapes(input_shapes):
    return [i for i in input_shapes.values() if not isinstance(i, tuple)][0][0]


def maybe_serialize_keras_objects(self, config, maybe_serialize_keys):
    for key in maybe_serialize_keys:
        maybe_value = getattr(self, key, None)
        if maybe_value:
            config[key] = tf.keras.utils.serialize_keras_object(maybe_value)

    return config


def maybe_deserialize_keras_objects(config, maybe_deserialize_keys):
    for key in maybe_deserialize_keys:
        if key in config:
            config[key] = tf.keras.utils.deserialize_keras_object(config[key])

    return config
