import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops

from ..utils.registry import Registry
from .typing import TensorOrTabularData

augmentation_registry: Registry = Registry.class_registry("tf.augmentation_registry")


class DataAugmentation(tf.keras.layers.Layer):
    def augment(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    def call(self, inputs: TensorOrTabularData, training=True, **kwargs) -> TensorOrTabularData:
        def augment():
            if isinstance(inputs, dict):
                return {key: self.augment(val) for key, val in inputs.items()}

            return self.augment(inputs)

        output = control_flow_util.smart_cond(training, augment, lambda: inputs)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape


@augmentation_registry.register_with_multiple_names("stochastic-swap-noise", "ssn")
class StochasticSwapNoise(DataAugmentation):
    """
    Applies Stochastic replacement of sequence features
    """

    def __init__(self, pad_token=0, replacement_prob=0.1):
        super().__init__()
        self.pad_token = pad_token
        self.replacement_prob = replacement_prob

    def augment(self, input_tensor: tf.Tensor, **kwargs) -> tf.Tensor:
        mask_matrix = backend.random_binomial(
            array_ops.shape(input_tensor), p=self.replacement_prob
        )
        swaps = tf.gather(input_tensor, tf.random.shuffle(tf.range(tf.shape(input_tensor)[0])))
        output = tf.where(
            (mask_matrix == 1) & (input_tensor != self.pad_token), swaps, input_tensor
        )

        return output
