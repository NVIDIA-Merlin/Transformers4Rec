from ..tabular import FilterFeatures, TabularModule


class ContinuousFeatures(TabularModule):
    def __init__(self, features, aggregation=None):
        super().__init__(aggregation)
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def forward(self, inputs):
        return self.filter_features(inputs)

    def forward_output_size(self, input_shapes):
        filtered = self.filter_features.forward_output_size(input_shapes)

        return super().forward_output_size(filtered)
