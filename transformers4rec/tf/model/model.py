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
from typing import List, Optional

import tensorflow as tf
from tensorflow.python.framework import ops

from ..utils.tf_utils import LossMixin
from .head import Head


class BaseModel(tf.keras.Model, LossMixin, abc.ABC):
    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            inputs, targets = inputs
            loss = self.compute_loss(inputs, targets, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss = self.compute_loss(*inputs, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


class Model(BaseModel):
    def __init__(
        self, *head: Head, head_weights: Optional[List[float]] = None, name=None, **kwargs
    ):
        if head_weights:
            if not isinstance(head_weights, (list, tuple)):
                raise ValueError("`head_weights` must be a list or tuple")
            if not len(head_weights) == len(head):
                raise ValueError(
                    "`head_weights` needs to have the same length " "as the number of heads"
                )

        super().__init__(name=name, **kwargs)

        self.heads = head
        self.head_weights = tuple(head_weights or [1.0] * len(head))

    def call(self, inputs, **kwargs):
        # TODO: Optimize this
        outputs = {}
        for head in self.heads:
            body_outputs = head.body(inputs)
            outputs.update(head(body_outputs, call_body=False, always_output_dict=True))

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(
        self, inputs, targets, training: bool = False, compute_metrics=True, **kwargs
    ) -> tf.Tensor:
        losses = tuple(
            [
                head.compute_loss(
                    inputs,
                    targets,
                    call_body=kwargs.pop("call_body", True),
                    compute_metrics=compute_metrics,
                    **kwargs
                )
                for head in self.heads
            ]
        )
        with ops.name_scope("merge_losses", values=losses + self.head_weights):
            weighted_losses = []
            for loss, head_weight in zip(losses, self.head_weights):
                weighted_losses.append(tf.math.multiply(loss, head_weight))

            return tf.add_n(weighted_losses)

    def metric_results(self, mode=None):
        outputs = []

        for head in self.heads:
            outputs.append(head.metric_results(mode=mode))

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    @classmethod
    def from_config(cls, config, custom_objects=None):
        heads = [tf.keras.utils.deserialize_keras_object(h) for h in config.pop("heads")]

        return cls(*heads, **config)

    def get_config(self):
        return {
            "head_weights": self.head_weights,
            "heads": [tf.keras.utils.serialize_keras_object(h) for h in self.heads],
        }
