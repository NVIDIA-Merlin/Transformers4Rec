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

import tensorflow as tf

_INTERACTION_TYPES = (None, "field_all", "field_each", "field_interaction")


class DotProductInteraction(tf.keras.layers.Layer):
    """
    Layer implementing the factorization machine style feature
    interaction layer suggested by the DLRM and DeepFM architectures,
    generalized to include a dot-product version of the parameterized
    interaction suggested by the FiBiNet architecture (which normally
    uses element-wise multiplication instead of dot product). Maps from
    tensors of shape `(batch_size, num_features, embedding_dim)` to
    tensors of shape `(batch_size, (num_features - 1)*num_features // 2)`
    if `self_interaction` is `False`, otherwise `(batch_size, num_features**2)`.

    Parameters
    ------------------------
    interaction_type: {}
        The type of feature interaction to use. `None` defaults to the
        standard factorization machine style interaction, and the
        alternatives use the implementation defined in the FiBiNet
        architecture (with the element-wise multiplication replaced
        with a dot product).
    self_interaction: bool
        Whether to calculate the interaction of a feature with itself.
    """.format(
        _INTERACTION_TYPES
    )

    def __init__(self, interaction_type=None, self_interaction=False, name=None, **kwargs):
        if interaction_type not in _INTERACTION_TYPES:
            raise ValueError("Unknown interaction type {}".format(interaction_type))
        self.interaction_type = interaction_type
        self.self_interaction = self_interaction
        super(DotProductInteraction, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.interaction_type is None:
            self.built = True
            return

        kernel_shape = [input_shape[2], input_shape[2]]
        if self.interaction_type in _INTERACTION_TYPES[2:]:
            idx = _INTERACTION_TYPES.index(self.interaction_type)
            for _ in range(idx - 1):
                kernel_shape.insert(0, input_shape[1])

        self.kernel = self.add_weight(
            name="bilinear_interaction_kernel",
            shape=kernel_shape,
            initializer="glorot_normal",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        right = inputs

        # first transform v_i depending on the interaction type
        if self.interaction_type is None:
            left = inputs
        elif self.interaction_type == "field_all":
            left = tf.matmul(inputs, self.kernel)
        elif self.interaction_type == "field_each":
            left = tf.einsum("b...k,...jk->b...j", inputs, self.kernel)
        else:
            left = tf.einsum("b...k,f...jk->bf...j", inputs, self.kernel)

        # do the interaction between v_i and v_j
        # output shape will be (batch_size, num_features, num_features)
        if self.interaction_type != "field_interaction":
            interactions = tf.matmul(left, right, transpose_b=True)
        else:
            interactions = tf.einsum("b...jk,b...k->b...j", left, right)

        # mask out the appropriate area
        ones = tf.reduce_sum(tf.zeros_like(interactions), axis=0) + 1
        mask = tf.linalg.band_part(ones, 0, -1)  # set lower diagonal to zero
        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)  # get rid of diagonal
        mask = tf.cast(mask, tf.bool)
        x = tf.boolean_mask(interactions, mask, axis=1)

        # masking destroys shape information, set explicitly
        x.set_shape(self.compute_output_shape(inputs.shape))
        return x

    def compute_output_shape(self, input_shape):
        if self.self_interaction:
            output_dim = input_shape[1] ** 2
        else:
            output_dim = input_shape[1] * (input_shape[1] - 1) // 2
        return (input_shape[0], output_dim)

    def get_config(self):
        return {
            "interaction_type": self.interaction_type,
            "self_interaction": self.self_interaction,
        }
