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


class XDeepFmOuterProduct(tf.keras.layers.Layer):
    """
    Layer implementing the outer product transformation used in
    the Compressed Interaction Network (CIN) proposed in
    in https://arxiv.org/abs/1803.05170. Treats the feature dimension
    H_k of a B x H_k x D feature embedding tensor as a feature map
    of the D embedding elements, and computes element-wise multiplication
    interaction between these maps and those from an initial input tensor
    x_0 before taking the inner product with a parameter matrix.

    Parameters
    ------------
    dim : int
      Feature dimension of the layer. Output will be of shape
      (batch_size, dim, embedding_dim)
    """

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def build(self, input_shapes):
        if not isinstance(input_shapes[0], (tuple, tf.TensorShape)):
            raise ValueError("Should be called on a list of inputs.")
        if len(input_shapes) != 2:
            raise ValueError("Should only have two inputs, found {}".format(len(input_shapes)))
        for shape in input_shapes:
            if len(shape) != 3:
                raise ValueError("Found shape {} without 3 dimensions".format(shape))
        if input_shapes[0][-1] != input_shapes[1][-1]:
            raise ValueError(
                "Last dimension should match, found dimensions {} and {}".format(
                    input_shapes[0][-1], input_shapes[1][-1]
                )
            )

        # H_k x H_{k-1} x m
        shape = (self.dim, input_shapes[0][1], input_shapes[1][1])
        self.kernel = self.add_weight(
            name="kernel", initializer="glorot_uniform", trainable=True, shape=shape
        )
        self.built = True

    def call(self, inputs):
        """
        Parameters
        ------------
        inputs : array-like(tf.Tensor)
          The two input tensors, the first of which should be the
          output of the previous layer, and the second of which
          should be the input to the CIN.
        """
        x_k_minus_1, x_0 = inputs

        # need to do shape manipulations so that we
        # can do element-wise multiply
        x_k_minus_1 = tf.expand_dims(x_k_minus_1, axis=2)  # B x H_{k-1} x 1 x D
        x_k_minus_1 = tf.tile(x_k_minus_1, [1, 1, x_0.shape[1], 1])  # B x H_{k-1} x m x D
        x_k_minus_1 = tf.transpose(x_k_minus_1, (1, 0, 2, 3))  # H_{k-1} x B x m x D
        z_k = x_k_minus_1 * x_0  # H_{k-1} x B x m x D
        z_k = tf.transpose(z_k, (1, 0, 2, 3))  # B x H_{k-1} x m x D

        # now we need to map to B x H_k x D
        x_k = tf.tensordot(self.kernel, z_k, axes=[[1, 2], [1, 2]])
        x_k = tf.transpose(x_k, (1, 0, 2))
        return x_k

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], self.dim, input_shapes[0][2])
