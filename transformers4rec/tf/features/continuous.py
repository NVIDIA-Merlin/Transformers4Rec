from typing import List, Optional

from .. import typing
from ..tabular.tabular import FilterFeatures
from .base import InputBlock


class ContinuousFeatures(InputBlock):
    def __init__(
        self,
        features: List[str],
        pre: Optional[typing.TabularTransformationType] = None,
        post: Optional[typing.TabularTransformationType] = None,
        aggregation: Optional[typing.TabularAggregationType] = None,
        name=None,
        **kwargs
    ):
        super().__init__(pre=pre, post=post, aggregation=aggregation, name=name, **kwargs)
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.filter_features(inputs)

    def compute_call_output_shape(self, input_shapes):
        return self.filter_features.compute_output_shape(input_shapes)

    def _get_name(self):
        return "ContinuousFeatures"

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]

    def repr_extra(self):
        return ", ".join(sorted(self.filter_features.to_include))
