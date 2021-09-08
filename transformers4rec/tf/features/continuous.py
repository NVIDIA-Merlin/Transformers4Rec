from typing import List, Optional

import tensorflow as tf

from ...types import DatasetSchema
from ...utils.misc_utils import docstring_parameter
from ..tabular.tabular import TABULAR_MODULE_PARAMS_DOCSTRING, FilterFeatures
from ..typing import TabularAggregationType, TabularTransformationType
from .base import InputBlock


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
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
        schema: Optional[DatasetSchema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.filter_features = FilterFeatures(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.filter_features(inputs)

    def compute_call_output_shape(self, input_shapes):
        return self.filter_features.compute_output_shape(input_shapes)

    def get_config(self):
        config = super().get_config()

        config["features"] = self.filter_features.to_include

        return config

    def _get_name(self):
        return "ContinuousFeatures"

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]

    def repr_extra(self):
        return ", ".join(sorted(self.filter_features.to_include))
