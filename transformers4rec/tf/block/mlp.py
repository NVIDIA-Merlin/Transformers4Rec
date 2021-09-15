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

from typing import List

import tensorflow as tf

from .base import SequentialBlock


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class MLPBlock(SequentialBlock):
    def __init__(
        self,
        dimensions: List[int],
        activation="relu",
        use_bias: bool = True,
        dropout=None,
        normalization=None,
        filter_features=None,
        **kwargs
    ):
        layers = []
        for dim in dimensions:
            layers.append(tf.keras.layers.Dense(dim, activation=activation, use_bias=use_bias))
            if dropout:
                layers.append(tf.keras.layers.Dropout(dropout))
            if normalization:
                if normalization == "batch_norm":
                    layers.append(tf.keras.layers.BatchNormalization())
                elif isinstance(normalization, tf.keras.layers.Layer):
                    layers.append(normalization)
                else:
                    raise ValueError(
                        "Normalization needs to be an instance `Layer` or " "`batch_norm`"
                    )

        super().__init__(layers, filter_features, **kwargs)
