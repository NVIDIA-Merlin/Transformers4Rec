from typing import Optional, Union

import torch

from ..types import DatasetSchema
from . import aggregation as agg
from . import augmentation as aug
from .typing import TabularData


class TabularModule(torch.nn.Module):
    """Torch Module that's specialized for tabular-data.

    Parameters
    ----------
    augmentation: Union[str, aug.DataAugmentation], optional
        Augmentation to apply on the inputs when the module is called.
    aggregation: Union[str, agg.FeatureAggregation], optional
        Aggregation to apply after processing the `forward`-method.
    """

    def __init__(
        self,
        augmentation: Optional[Union[str, aug.DataAugmentation]] = None,
        aggregation: Optional[Union[str, agg.FeatureAggregation]] = None,
    ):
        super().__init__()
        self.input_size = None
        self.augmentation = augmentation
        self.aggregation = aggregation

    @property
    def aggregation(self):
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: Optional[Union[str, agg.FeatureAggregation]]):
        if value:
            self._aggregation = agg.aggregation_registry.parse(value)
        else:
            self._aggregation = None

    @property
    def augmentation(self):
        return self._augmentation

    @augmentation.setter
    def augmentation(self, value: Optional[Union[str, aug.DataAugmentation]]):
        if value:
            self._augmentation = aug.augmentation_registry.parse(value)
        else:
            self._augmentation = None

    @classmethod
    def from_schema(cls, schema: DatasetSchema, tags=None, **kwargs) -> Optional["TabularModule"]:
        """

        Parameters
        ----------
        schema
        tags
        kwargs

        Returns
        -------

        """
        if tags:
            schema = schema.select_by_tag(tags)

        if not schema.columns:
            return None

        return cls.from_features(schema.column_names, **kwargs)

    @classmethod
    def from_features(cls, features, **kwargs):
        from .block.base import SequentialBlock

        return SequentialBlock(features, **kwargs)

    def forward(self, x: TabularData, *args, **kwargs) -> TabularData:
        return x

    def __call__(
        self,
        inputs,
        *args,
        pre=None,
        post=None,
        merge_with=None,
        stack_outputs=False,
        concat_outputs=False,
        aggregation=None,
        augmentation=None,
        filter_columns=None,
        **kwargs
    ):
        """

        Parameters
        ----------
        inputs
        args
        pre
        post
        merge_with
        stack_outputs
        concat_outputs
        aggregation
        augmentation
        filter_columns
        kwargs

        Returns
        -------

        """
        augmentation = augmentation or getattr(self, "augmentation", None)
        post_op = getattr(self, "aggregation", None)
        if concat_outputs:
            post_op = agg.ConcatFeatures()
        if stack_outputs:
            post_op = agg.StackFeatures()
        if aggregation:
            post_op = agg.aggregation_registry.parse(aggregation)

        if filter_columns:
            pre = FilterFeatures(filter_columns)
        if pre:
            inputs = pre(inputs)

        if augmentation:
            augmentation = aug.augmentation_registry.parse(augmentation)
            inputs = augmentation(inputs)

        outputs = super().__call__(inputs, *args, **kwargs)  # noqa

        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        if isinstance(outputs, dict) and post_op:
            outputs = post_op(outputs)

        return outputs


class FilterFeatures(torch.nn.Module):
    def __init__(self, to_include, pop=False):
        super().__init__()
        self.to_include = to_include
        self.pop = pop

    def forward(self, inputs: TabularData) -> TabularData:
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.to_include}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def forward_output_size(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.to_include}


def merge_tabular(self, other, **kwargs):
    from .block.base import MergeTabular

    return MergeTabular(self, other)


TabularModule.__add__ = merge_tabular
TabularModule.merge = merge_tabular
