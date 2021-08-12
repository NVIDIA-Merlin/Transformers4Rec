from functools import reduce
from typing import Optional

import torch

from ..types import Schema
from . import aggregation as agg


class TabularMixin:
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
        filter_columns=None,
        **kwargs
    ):
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
        outputs = super().__call__(inputs, *args, **kwargs)  # noqa

        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        if post_op:
            outputs = post_op(outputs)

        return outputs


class FilterFeatures(TabularMixin, torch.nn.Module):
    def __init__(self, to_include, pop=False):
        super().__init__()
        self.to_include = to_include
        self.pop = pop

    def forward(self, inputs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.to_include}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def forward_output_size(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.to_include}


class AsTabular(torch.nn.Module):
    def __init__(self, output_name):
        super().__init__()
        self.output_name = output_name

    def forward(self, inputs):
        return {self.output_name: inputs}

    def build(self, input_size, device=None):
        if device:
            self.to(device)
        self.input_size = input_size

        return self

    def forward_output_size(self, input_size):
        return {self.output_name: input_size}

    def output_size(self):
        if not self.input_size:
            # TODO: log warning here
            pass

        return self.forward_output_size(self.input_size)


class TabularModule(TabularMixin, torch.nn.Module):
    def __init__(self, aggregation=None):
        super().__init__()
        self.input_size = None
        self.aggregation = aggregation

    @property
    def aggregation(self):
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value):
        if value:
            self._aggregation = agg.aggregation_registry.parse(value)
        else:
            self._aggregation = None

    @classmethod
    def from_schema(cls, schema: Schema, tags=None, **kwargs) -> Optional["TabularModule"]:
        if tags:
            schema = schema.select_by_tag(tags)

        if not schema.columns:
            return None

        return cls.from_features(schema.column_names, **kwargs)

    @classmethod
    def from_features(cls, features, **kwargs):
        return features >> cls(**kwargs)

    def forward(self, x, *args, **kwargs):
        return x

    def build(self, input_size, device=None):
        if device:
            self.to(device)
        self.input_size = input_size

        return self

    def forward_output_size(self, input_size):
        if self.aggregation:
            return self.aggregation.forward_output_size(input_size)

        return input_size

    def output_size(self):
        if not self.input_size:
            # TODO: log warning here
            pass

        return self.forward_output_size(self.input_size)

    def __rrshift__(self, other):
        from .block.base import right_shift_block

        return right_shift_block(self, other)


class MergeTabular(TabularModule):
    def __init__(self, *to_merge, aggregation=None):
        super().__init__(aggregation)
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

    def forward(self, inputs):
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

    def build(self, input_size, device=None):
        super().build(input_size, device)

        for layer in self.merge_values:
            layer.build(input_size, device)

        return self


def merge_tabular(self, other, **kwargs):
    return MergeTabular(self, other)


TabularModule.__add__ = merge_tabular
TabularModule.merge = merge_tabular
