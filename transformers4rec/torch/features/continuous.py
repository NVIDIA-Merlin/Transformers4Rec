from typing import List, Optional

from ...utils.misc_utils import docstring_parameter
from .. import typing
from ..tabular.tabular import TABULAR_MODULE_PARAMS_DOCSTRING, FilterFeatures
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
        pre: Optional[typing.TabularTransformationType] = None,
        post: Optional[typing.TabularTransformationType] = None,
        aggregation: Optional[typing.TabularAggregationType] = None,
    ):
        super().__init__(aggregation=aggregation, pre=pre, post=post)
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def forward(self, inputs, **kwargs):
        return self.filter_features(inputs)

    def forward_output_size(self, input_sizes):
        return self.filter_features.forward_output_size(input_sizes)
