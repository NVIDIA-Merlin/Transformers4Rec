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

import torch

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
        **kwargs
    ):
        super().__init__(aggregation=aggregation, pre=pre, post=post, schema=schema)
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def forward(self, inputs, **kwargs):
        cont_features = self.filter_features(inputs)
        cont_features = {k: v.unsqueeze(-1) for k, v in cont_features.items()}
        return cont_features

    def forward_output_size(self, input_sizes):
        cont_features_sizes = self.filter_features.forward_output_size(input_sizes)
        cont_features_sizes = {k: torch.Size(list(v) + [1]) for k, v in cont_features_sizes.items()}
        return cont_features_sizes
