from ..tabular import FilterFeatures
from .base import InputModule
from .normalization import LayerNormalizationFeatures


class ContinuousFeatures(InputModule):
    def __init__(self, features, layer_norm: bool = True, aggregation=None, augmentation=None):
        super().__init__(aggregation=aggregation, augmentation=augmentation)
        self.filter_features = FilterFeatures(features)

        features_dim = {fname: 1 for fname in features}
        if layer_norm:
            self.layer_norm_features = LayerNormalizationFeatures(features_dim)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def forward(self, inputs, **kwargs):
        cont_features = self.filter_features(inputs)
        if self.layer_norm_features:
            # Adding an additional dim so that continuous features have its last dim=1
            cont_features = {fname: cont_features[fname].unsqueeze(-1) for fname in cont_features}
            cont_features_norm = self.layer_norm_features(cont_features)
            # Removing the temporary last dim added after normalization
            cont_features = {
                fname: cont_features[fname].squeeze(-1) for fname in cont_features_norm
            }
        return cont_features

    def forward_output_size(self, input_shapes):
        filtered = self.filter_features.forward_output_size(input_shapes)

        return super().forward_output_size(filtered)
