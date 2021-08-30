from ..block.tabular.tabular import FilterFeatures
from .base import InputBlock


class ContinuousFeatures(InputBlock):
    def __init__(self, features, aggregation=None, pre=None, post=None, **kwargs):
        super().__init__(aggregation=aggregation, pre=pre, post=post)
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def forward(self, inputs, **kwargs):
        return self.filter_features(inputs)

    def forward_output_size(self, input_sizes):
        return self.filter_features.forward_output_size(input_sizes)
