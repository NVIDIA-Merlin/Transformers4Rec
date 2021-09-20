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

from typing import Optional

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops

from ..typing import TabularData, TensorOrTabularData
from .base import TabularTransformation, tabular_transformation_registry


@tabular_transformation_registry.register("as-sparse")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class AsSparseFeatures(TabularTransformation):
    def call(self, inputs: TabularData, **kwargs) -> TabularData:
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
        return input_shape


@tabular_transformation_registry.register("as-dense")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class AsDenseFeatures(TabularTransformation):
    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_tensor()
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tabular_transformation_registry.register_with_multiple_names("stochastic-swap-noise", "ssn")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class StochasticSwapNoise(TabularTransformation):
    """
    Applies Stochastic replacement of sequence features
    """

    def __init__(self, schema=None, pad_token=0, replacement_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema
        self.pad_token = pad_token
        self.replacement_prob = replacement_prob

    def call(
        self,
        inputs: TensorOrTabularData,
        input_mask: Optional[tf.Tensor] = None,
        training=True,
        **kwargs
    ) -> TensorOrTabularData:
        def augment(input_mask):
            if self.schema:
                input_mask = input_mask or self.get_padding_mask_from_item_id(
                    inputs, self.pad_token
                )

            if isinstance(inputs, dict):
                return {key: self.augment(val, input_mask) for key, val in inputs.items()}

            return self.augment(inputs, input_mask)

        output = control_flow_util.smart_cond(training, lambda: augment(input_mask), lambda: inputs)

        return output

    def augment(self, input_tensor: tf.Tensor, mask: Optional[tf.Tensor], **kwargs) -> tf.Tensor:
        if mask is not None:
            if len(input_tensor.shape) == len(mask.shape) - 1:
                mask = mask[:, 0]

        replacement_mask_matrix = (
            tf.cast(
                backend.random_binomial(array_ops.shape(input_tensor), p=self.replacement_prob),
                tf.int32,
            )
            * tf.cast(mask, tf.int32)
        )

        n_values_to_replace = tf.reduce_sum(replacement_mask_matrix)

        input_flattened_non_zero = tf.boolean_mask(
            input_tensor, tf.cast(replacement_mask_matrix, tf.bool)
        )

        sampled_values_to_replace = tf.gather(
            input_flattened_non_zero,
            tf.random.shuffle(tf.range(tf.shape(input_flattened_non_zero)[0]))[
                :n_values_to_replace
            ],
        )

        replacement_indices = tf.sparse.from_dense(replacement_mask_matrix).indices

        output_tensor = tf.tensor_scatter_nd_update(
            input_tensor, replacement_indices, sampled_values_to_replace
        )

        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config["pad_token"] = self.pad_token
        config["replacement_prob"] = self.replacement_prob

        return config
