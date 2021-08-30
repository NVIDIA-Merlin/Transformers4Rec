from typing import Dict, Optional

import torch

from ...typing import FeatureConfig, TableConfig, TabularData, TensorOrTabularData
from .tabular import TabularTransformation, tabular_transformation_registry


@tabular_transformation_registry.register_with_multiple_names("stochastic-swap-noise", "ssn")
class StochasticSwapNoise(TabularTransformation):
    """
    Applies Stochastic replacement of sequence features
    """

    def __init__(self, schema=None, pad_token=0, replacement_prob=0.1):
        super().__init__()
        self.schema = schema
        self.pad_token = pad_token
        self.replacement_prob = replacement_prob

    def forward(self, inputs: TensorOrTabularData, **kwargs) -> TensorOrTabularData:
        maybe_mask = self.get_mask(inputs, self.pad_token)
        if isinstance(inputs, dict):
            return {key: self.augment(val, maybe_mask) for key, val in inputs.items()}

        return self.augment(inputs, maybe_mask)

    def forward_output_size(self, input_size):
        return input_size

    def augment(
        self, input_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # TODO: make this work when `mask = None`
        with torch.no_grad():
            sse_prob_replacement_matrix = torch.full(
                input_tensor.shape,
                self.replacement_prob,
                device=input_tensor.device,
            )
            sse_replacement_mask = torch.bernoulli(sse_prob_replacement_matrix).bool()
            if mask is not None:
                sse_replacement_mask = sse_replacement_mask & mask
            n_values_to_replace = sse_replacement_mask.sum()

            cdata_flattened_non_zero = torch.masked_select(input_tensor, mask)

            sampled_values_to_replace = cdata_flattened_non_zero[
                torch.randperm(cdata_flattened_non_zero.shape[0])
            ][:n_values_to_replace]

            input_tensor[sse_replacement_mask] = sampled_values_to_replace

        return input_tensor


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
