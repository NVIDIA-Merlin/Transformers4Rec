from typing import List

from ..tabular import FilterFeatures
from .base import InputLayer


class ContinuousFeatures(InputLayer):
    def __init__(
        self,
        features,
        aggregation=None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(aggregation, trainable, name, dtype, dynamic, **kwargs)
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.filter_features(inputs)

    def compute_output_shape(self, input_shapes):
        filtered = self.filter_features.compute_output_shape(input_shapes)

        return super(ContinuousFeatures, self).compute_output_shape(filtered)

    def _get_name(self):
        return "ContinuousFeatures"

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]

    def repr_extra(self):
        return ", ".join(sorted(self.filter_features.columns))
