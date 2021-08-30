from abc import ABC
from functools import reduce
from typing import List, Optional, Union

import torch

from ....types import DatasetSchema
from ....utils.registry import Registry
from ...typing import TabularData, TensorOrTabularData
from ...utils.torch_utils import OutputSizeMixin, SchemaMixin
from ..base import BlockBase, SequentialBlock, right_shift_block

tabular_transformation_registry: Registry = Registry.class_registry("torch.tabular_transformations")
tabular_aggregation_registry: Registry = Registry.class_registry("torch.tabular_aggregations")


class TabularTransformation(torch.nn.Module, OutputSizeMixin, SchemaMixin, ABC):
    def forward(self, inputs: TabularData, **kwargs) -> TabularData:
        raise NotImplementedError()

    @classmethod
    def parse(cls, class_or_str):
        return tabular_transformation_registry.parse(class_or_str)


class TabularAggregation(torch.nn.Module, OutputSizeMixin, SchemaMixin, ABC):
    def forward(self, inputs: TabularData) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def parse(cls, class_or_str):
        return tabular_aggregation_registry.parse(class_or_str)


class TabularTransformations(SequentialBlock):
    def __init__(self, *transformation):
        if len(transformation) == 1 and isinstance(transformation, list):
            transformation = transformation[0]
        super().__init__(*[TabularTransformation.parse(t) for t in transformation])

    def append(self, transformation):
        self.transformations.append(TabularTransformation.parse(transformation))


Transformations = Union[str, TabularTransformation, List[str], List[TabularTransformation]]
Aggregation = Union[str, TabularAggregation]


class TabularModule(torch.nn.Module):
    """PyTorch Module that's specialized for tabular-data.

    Parameters
    ----------
    augmentation: Union[str, aug.DataAugmentation], optional
        Augmentation to apply on the inputs when the module is called.
    aggregation: Union[str, agg.FeatureAggregation], optional
        Aggregation to apply after processing the `forward`-method.
    """

    def __init__(
        self,
        pre: Optional[Transformations] = None,
        post: Optional[Transformations] = None,
        aggregation: Optional[Aggregation] = None,
        **kwargs
    ):
        super().__init__()
        self.input_size = None
        self.pre = pre
        self.post = post
        self.aggregation = aggregation

    @property
    def pre(self) -> Optional[TabularTransformations]:
        """

        Returns
        -------

        """
        return self._pre

    @pre.setter
    def pre(self, value: Union[str, TabularTransformation, List[str], List[TabularTransformation]]):
        if value:
            self._pre = TabularTransformations(value)
        else:
            self._pre = None

    @property
    def post(self) -> Optional[TabularTransformations]:
        """

        Returns
        -------

        """
        return self._post

    @post.setter
    def post(
        self, value: Union[str, TabularTransformation, List[str], List[TabularTransformation]]
    ):
        if value:
            self._post = TabularTransformations(value)
        else:
            self._post = None

    @property
    def aggregation(self) -> Optional[TabularAggregation]:
        """

        Returns
        -------

        """
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: Optional[Union[str, TabularAggregation]]):
        """

        Parameters
        ----------
        value
        """
        if value:
            self._aggregation = TabularAggregation.parse(value)
        else:
            self._aggregation = None

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
        return SequentialBlock(features, **kwargs)

    def pre_forward(
        self, inputs: TabularData, transformations: Optional[Transformations] = None
    ) -> TabularData:
        return self._maybe_apply_transformations(inputs, "pre", transformations=transformations)

    def forward(self, x: TabularData, *args, **kwargs) -> TabularData:
        return x

    def post_forward(
        self,
        inputs: TabularData,
        transformations: Optional[Transformations] = None,
        merge_with: Union["TabularModule", List["TabularModule"]] = None,
        aggregation: Optional[Aggregation] = None,
    ) -> TensorOrTabularData:
        if aggregation:
            aggregation = TabularAggregation.parse(aggregation)
        aggregation = aggregation or getattr(self, "aggregation", None)

        outputs = inputs
        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        outputs = self._maybe_apply_transformations(
            outputs, "post", transformations=transformations
        )

        if aggregation:
            return aggregation(outputs)

        return outputs

    def __call__(
        self,
        inputs: TabularData,
        *args,
        pre: Optional[Transformations] = None,
        post: Optional[Transformations] = None,
        merge_with: Union["TabularModule", List["TabularModule"]] = None,
        aggregation: Optional[Aggregation] = None,
        **kwargs
    ):
        """

        Parameters
        ----------
        inputs
        pre
        post
        merge_with
        aggregation

        Returns
        -------

        """
        inputs = self.pre_forward(inputs, transformations=pre)

        # This will call the `forward` method implemented by the super class.
        outputs = super().__call__(inputs, *args, **kwargs)  # noqa

        if isinstance(outputs, dict):
            outputs = self.post_forward(
                outputs, transformations=post, merge_with=merge_with, aggregation=aggregation
            )

        return outputs

    def _maybe_apply_transformations(
        self, inputs: TabularData, key: str, transformations: Optional[Transformations] = None
    ) -> TabularData:
        if transformations:
            transformations = TabularTransformation.parse(transformations)
        transformations = transformations or getattr(self, key, None)

        if transformations:
            return transformations(inputs)

        return inputs

    def __rrshift__(self, other):
        return right_shift_block(self, other)


class FilterFeatures(TabularTransformation):
    """Module that filters out certain features from `TabularData`."

    Parameters
    ----------
    to_include: List[str]
        List of features to include in the result of calling the module
    pop: bool
        Boolean indicating whether to pop the features to exclude from the inputs dictionary.
    """

    def __init__(self, to_include: List[str], pop: bool = False):
        super().__init__()
        self.to_include = to_include
        self.pop = pop

    def forward(self, inputs: TabularData, **kwargs) -> TabularData:
        """

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing features to filter.

        Returns Filtered TabularData that only contains the feature-names in `self.to_include`.
        -------

        """
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.to_include}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def forward_output_size(self, input_shape):
        """

        Parameters
        ----------
        input_shape

        Returns
        -------

        """
        return {k: v for k, v in input_shape.items() if k in self.to_include}


class TabularBlock(BlockBase, TabularModule, ABC):
    def to_module(self, shape_or_module, device=None):
        shape = shape_or_module
        if isinstance(shape_or_module, torch.nn.Module):
            shape = getattr(shape_or_module, "output_size", None)
            if shape:
                shape = shape()

        return self.build(shape, device=device)

    def output_size(self, input_size=None):
        if self.pre:
            output_size = self.pre.output_size(input_size)

        output_size = self._check_post_output_size(super().output_size(input_size))

        return output_size

    def build(self, input_size, **kwargs):
        if self.pre:
            self.pre.build(input_size, **kwargs)

        if self.post:
            self.post.build(input_size, **kwargs)

        if self.aggregation:
            self.aggregation.build(input_size, **kwargs)

        return super().build(input_size, **kwargs)

    def _check_post_output_size(self, input_size):
        output_size = input_size

        if isinstance(input_size, dict):
            if self.post:
                output_size = self.post.output_size(output_size)
            if self.aggregation:
                output_size = self.aggregation.forward_output_size(output_size)

        return output_size

    def __rrshift__(self, other):
        return right_shift_block(self, other)


class MergeTabular(TabularBlock):
    def __init__(self, *to_merge, pre=None, post=None, aggregation=None):
        super().__init__(pre=pre, post=post, aggregation=aggregation)
        if all(isinstance(x, dict) for x in to_merge):
            to_merge = reduce(lambda a, b: dict(a, **b), to_merge)
            self.to_merge = torch.nn.ModuleDict(to_merge)
        else:
            self.to_merge = torch.nn.ModuleList(to_merge)

    @property
    def merge_values(self):
        if isinstance(self.to_merge, torch.nn.ModuleDict):
            return list(self.to_merge.values())

        return self.to_merge

    def forward(self, inputs: TabularData, **kwargs) -> TabularData:
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        for layer in self.merge_values:
            outputs.update(layer(inputs))

        return outputs

    def forward_output_size(self, input_size):
        output_shapes = {}

        for layer in self.merge_values:
            output_shapes.update(layer.forward_output_size(input_size))

        return super(MergeTabular, self).forward_output_size(output_shapes)

    def build(self, input_size, **kwargs):
        super().build(input_size, **kwargs)

        for layer in self.merge_values:
            layer.build(input_size, **kwargs)

        return self


class AsTabular(TabularBlock):
    def __init__(self, output_name):
        super().__init__()
        self.output_name = output_name

    def forward(self, inputs, **kwargs) -> TabularData:
        return {self.output_name: inputs}

    def forward_output_size(self, input_size):
        return {self.output_name: input_size}


def merge_tabular(self, other, **kwargs):
    return MergeTabular(self, other)


TabularModule.__add__ = merge_tabular
TabularModule.merge = merge_tabular
