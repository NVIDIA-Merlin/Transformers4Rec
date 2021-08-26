from abc import ABC
from typing import List, Optional

import tensorflow as tf

from .head import Head
from .typing import LossReduction


class ModelWithLoss(tf.keras.Model, ABC):
    def compute_loss(self, inputs, targets, training: bool = False, **kwargs) -> tf.Tensor:
        raise NotImplementedError("Sub-classes must implement the `compute_loss` method.")

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


class Model(ModelWithLoss):
    def __init__(
        self,
        *head: Head,
        head_weights: Optional[List[float]] = None,
        head_reduction: Optional[LossReduction] = tf.reduce_mean,
        name=None,
        **kwargs
    ):
        if head_weights:
            if not isinstance(head_weights, list):
                raise ValueError("`head_weights` must be a list")
            if not len(head_weights) == len(head):
                raise ValueError(
                    "`head_weights` needs to have the same length " "as the number of heads"
                )

        super().__init__(name=name, **kwargs)

        self.heads = head
        self.head_weights = head_weights or [1.0] * len(head)
        self.head_reduction = head_reduction

    def call(self, inputs, **kwargs):
        # TODO: Optimize this
        outputs = {}
        for head in self.heads:
            outputs.update(head(inputs, call_body=True, always_output_dict=True))

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(
        self, inputs, targets, training: bool = False, compute_metrics=True, **kwargs
    ) -> tf.Tensor:
        losses = []

        for i, head in enumerate(self.heads):
            loss = head.compute_loss(
                inputs, targets, call_body=True, compute_metrics=compute_metrics, **kwargs
            )
            losses.append(loss * self.head_weights[i])

        losses_tensor = tf.concat(losses, axis=0)

        return self.head_reduction(losses_tensor)

    def get_config(self):
        pass
