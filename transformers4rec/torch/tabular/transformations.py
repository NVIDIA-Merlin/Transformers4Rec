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

from typing import Dict, Optional

import torch

from ..features.embedding import FeatureConfig, TableConfig
from ..typing import TabularData, TensorOrTabularData
from .base import TabularTransformation, tabular_transformation_registry


@tabular_transformation_registry.register_with_multiple_names("stochastic-swap-noise", "ssn")
class StochasticSwapNoise(TabularTransformation):
    """
    Applies Stochastic replacement of sequence features.
    It can be applied as a `pre` transform like `TransformerBlock(pre="stochastic-swap-noise")`
    """

    def __init__(self, schema=None, pad_token=0, replacement_prob=0.1):
        super().__init__()
        self.schema = schema
        self.pad_token = pad_token
        self.replacement_prob = replacement_prob

    def forward(  # type: ignore
        self, inputs: TensorOrTabularData, input_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> TensorOrTabularData:
        if self.schema:
            input_mask = input_mask or self.get_padding_mask_from_item_id(inputs, self.pad_token)
        if isinstance(inputs, dict):
            return {key: self.augment(val, input_mask) for key, val in inputs.items()}

        return self.augment(inputs, input_mask)

    def forward_output_size(self, input_size):
        return input_size

    def augment(
        self, input_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            if mask is not None:
                if input_tensor.ndim == mask.ndim - 1:
                    mask = mask[:, 0]

            sse_prob_replacement_matrix = torch.full(
                input_tensor.shape,
                self.replacement_prob,
                device=input_tensor.device,
            )
            sse_replacement_mask = torch.bernoulli(sse_prob_replacement_matrix).bool()
            if mask is not None:
                sse_replacement_mask = sse_replacement_mask & mask
            n_values_to_replace = sse_replacement_mask.sum()

            if mask is not None:
                masked = torch.masked_select(input_tensor, mask)
            else:
                masked = torch.clone(input_tensor)

            input_permutation = torch.randperm(masked.shape[0])
            sampled_values_to_replace = masked[input_permutation][
                :n_values_to_replace  # type: ignore
            ]
            output_tensor = input_tensor.clone()

            if input_tensor[sse_replacement_mask].size() != sampled_values_to_replace:
                sampled_values_to_replace = torch.squeeze(sampled_values_to_replace)

            output_tensor[sse_replacement_mask] = sampled_values_to_replace

        return output_tensor


@tabular_transformation_registry.register_with_multiple_names("layer-norm")
class TabularLayerNorm(TabularTransformation):
    """
    Applies Layer norm to each input feature individually, before the aggregation
    """

    def __init__(self, features_dim: Optional[Dict[str, int]] = None):
        super().__init__()
        feature_layer_norm = {}
        if features_dim:
            for fname, dim in features_dim.items():
                feature_layer_norm[fname] = torch.nn.LayerNorm(normalized_shape=dim)
        self.feature_layer_norm = torch.nn.ModuleDict(feature_layer_norm)

    @classmethod
    def from_feature_config(cls, feature_config: Dict[str, FeatureConfig]):
        features_dim = {}
        for name, feature in feature_config.items():
            table: TableConfig = feature.table
            features_dim[name] = table.dim
        return cls(features_dim)

    def forward(self, inputs: TabularData, **kwargs) -> TabularData:
        if len(self.feature_layer_norm) == 0:
            raise ValueError("`features_dim is empty.`")
        return {key: self.feature_layer_norm[key](val) for key, val in inputs.items()}

    def forward_output_size(self, input_size):
        return input_size

    def build(self, input_size, **kwargs):
        if len(self.feature_layer_norm) == 0:
            for key, size in input_size.items():
                self.feature_layer_norm[key] = torch.nn.LayerNorm(normalized_shape=size[-1])

        return super().build(input_size, **kwargs)
