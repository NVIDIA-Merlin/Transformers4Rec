import tensorflow as tf

from ..head import Head
from .base import BlockType


class ModelWithLoss(tf.keras.Model):
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


class BlockWithHead(ModelWithLoss):
    def __init__(self, block: BlockType, head: Head, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.block = block
        self.head = head

    def call(self, inputs, **kwargs):
        return self.head(self.block(inputs, **kwargs), **kwargs)

    def compute_loss(self, inputs, targets, training: bool = False, **kwargs) -> tf.Tensor:
        block_outputs = self.block(inputs, training=training)

        return self.head.compute_loss(block_outputs, targets, training=training, **kwargs)

    def _get_name(self):
        return self.model_name if self.model_name else f"{self.block.__class__.__name__}WithHead"

    def get_config(self):
        pass
