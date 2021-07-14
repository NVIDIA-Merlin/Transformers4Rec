import tensorflow as tf

from .base import BlockType
from ..heads import Head


class ModelWithLoss(tf.keras.Model):
    """Base model for models.

    TODO: This is directly ported from tensorflow recommenders, should we depend on that?

    Many recommender models are relatively complex, and do not neatly fit into
    supervised or unsupervised paradigms. This base class makes it easy to
    define custom training and test losses for such complex models.
    This is done by asking the user to implement the following methods:
    - `__init__` to set up your model. Variable, task, loss, and metric
      initialization should go here.
    - `compute_loss` to define the training loss. The method takes as input the
      raw features passed into the model, and returns a loss tensor for training.
      As part of doing so, it should also update the model's metrics.
    - [Optional] `call` to define how the model computes its predictions. This
      is not always necessary: for example, two-tower retrieval models have two
      well-defined submodels whose `call` methods are normally used directly.
    Note that this base class is a thin conveniece wrapper for tf.keras.Model, and
    equivalent functionality can easily be achieved by overriding the `train_step`
    and `test_step` methods of a plain Keras model. Doing so also makes it easy
    to build even more complex training mechanisms, such as the use of
    different optimizers for different variables, or manipulating gradients.
    Keras has an excellent tutorial on how to
    do this [here](
    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
    """

    def compute_loss(self, inputs, targets, training: bool = False, **kwargs) -> tf.Tensor:
        """Defines the loss function.
        Args:
          inputs: A data structure of tensors: raw inputs to the model. These will
            usually contain labels and weights as well as features.
          training: Whether the model is in training mode.
        Returns:
          Loss tensor.
        """

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
    def __init__(self, block: BlockType, head: Head, model_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
