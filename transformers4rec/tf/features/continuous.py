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

from typing import List, Optional

import tensorflow as tf

from merlin_standard_lib import Schema
from merlin_standard_lib.utils.doc_utils import docstring_parameter

from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    FilterFeatures,
    TabularAggregationType,
    TabularTransformationType,
)
from .base import InputBlock


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class ContinuousFeatures(InputBlock):
    """Input block for continuous features.

    Parameters
    ----------
    features: List[str]
        List of continuous features to include in this module.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        features: List[str],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def call(self, inputs, *args, **kwargs):
        cont_features = self.filter_features(inputs)
        cont_features = {k: tf.expand_dims(v, -1) for k, v in cont_features.items()}
        return cont_features

    def compute_call_output_shape(self, input_shapes):
        cont_features_sizes = self.filter_features.compute_output_shape(input_shapes)
        cont_features_sizes = {
            k: tf.TensorShape(list(v) + [1]) for k, v in cont_features_sizes.items()
        }
        return cont_features_sizes

    def get_config(self):
        config = super().get_config()

        config["features"] = self.filter_features.to_include

        return config

    def _get_name(self):
        return "ContinuousFeatures"

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]

    def repr_extra(self):
        return ", ".join(sorted(self.filter_features.to_include))
