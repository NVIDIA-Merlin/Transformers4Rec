from typing import Dict

import torch

from ..typing import TabularData


class LayerNormalizationFeatures(torch.nn.Module):
    """
    Applies Layer norm to each input feature individually, before the aggregation
    """

    def __init__(self, features_dim: Dict[str, int]):
        super().__init__()

        feature_layer_norm = {}

        for fname, dim in features_dim.items():
            feature_layer_norm[fname] = torch.nn.LayerNorm(normalized_shape=dim)

        self.feature_layer_norm = torch.nn.ModuleDict(feature_layer_norm)

    def forward(self, inputs: TabularData) -> TabularData:
        return {key: self.feature_layer_norm[key](val) for key, val in inputs.items()}
