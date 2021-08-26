from typing import List

import tensorflow as tf

from .base import SequentialBlock


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
