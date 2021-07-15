from typing import Optional

import torch

from ..types import ColumnGroup


class FilterFeatures(torch.nn.Module):
    def __init__(self, columns, pop=False):
        super().__init__()
        self.columns = columns
        self.pop = pop

    def forward(self, inputs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.columns}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def forward_output_shape(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.columns}


class ConcatFeatures(torch.nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])

        return torch.cat(tensors, dim=self.axis)


class StackFeatures(torch.nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])

        return torch.stack(tensors, dim=self.axis)


class AsTabular(torch.nn.Module):
    def __init__(self, output_name):
        super().__init__()
        self.output_name = output_name

    def forward(self, inputs):
        return {self.output_name: inputs}


class TabularMixin:
    def set_aggregation(self, aggregation):
        if aggregation == "concat":
            self.aggregation = ConcatFeatures()
        elif aggregation == "stack":
            self.aggregation = StackFeatures()
        else:
            self.aggregation = None

    def __call__(
        self,
        inputs,
        *args,
        pre=None,
        post=None,
        merge_with=None,
        stack_outputs=False,
        concat_outputs=False,
        filter_columns=None,
        **kwargs
    ):
        post_op = getattr(self, "aggregation", None)
        if concat_outputs:
            post_op = ConcatFeatures()
        if stack_outputs:
            post_op = StackFeatures()
        if filter_columns:
            pre = FilterFeatures(filter_columns)
        if pre:
            inputs = pre(inputs)
        outputs = super().__call__(inputs, *args, **kwargs)

        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        if post_op:
            outputs = post_op(outputs)

        return outputs


class TabularModule(TabularMixin, torch.nn.Module):
    def __init__(self, aggregation=None):
        super().__init__()
        self.set_aggregation(aggregation)
        self.input_size = None

    @classmethod
    def from_column_group(
        cls, column_group: ColumnGroup, tags=None, tags_to_filter=None, **kwargs
    ) -> Optional["TabularModule"]:
        if tags:
            column_group = column_group.get_tagged(tags, tags_to_filter=tags_to_filter)

        if not column_group.columns:
            return None

        return cls.from_features(column_group.columns, **kwargs)

    @classmethod
    def from_schema(
        cls, schema, tags=None, tags_to_filter=None, **kwargs
    ) -> Optional["TabularModule"]:
        from nvtabular.column_group import ColumnGroup

        col_group = ColumnGroup.from_schema(schema)

        return cls.from_column_group(col_group, tags=tags, tags_to_filter=tags_to_filter, **kwargs)

    @classmethod
    def from_features(cls, features, **kwargs):
        return features >> cls(**kwargs)

    def forward(self, x, *args, **kwargs):
        return x

    def build(self, input_size, device=None):
        if device:
            self.to(device)
        self.input_size = input_size

    def forward_output_size(self, input_size):
        batch_size = self.calculate_batch_size_from_input_size(input_size)
        if isinstance(self.aggregation, ConcatFeatures):
            return batch_size, sum([i[1] for i in input_size.values()])
        elif isinstance(self.aggregation, StackFeatures):
            last_dim = [i for i in input_size.values()][0][-1]

            return batch_size, len(input_size), last_dim

        return input_size

    def output_size(self):
        if not self.input_size:
            # TODO: log warning here
            pass

        return self.forward_output_size(self.input_size)

    def calculate_batch_size_from_input_size(self, input_size):
        return [i for i in input_size.values() if isinstance(i, torch.Size)][0][0]

    def __rrshift__(self, other):
        from merlin_models.torch import right_shift_block

        return right_shift_block(self, other)


class MergeTabular(TabularModule):
    def __init__(self, *to_merge, aggregation=None):
        self.to_merge = list(to_merge)
        super().__init__(aggregation)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        for layer in self.to_merge:
            outputs.update(layer(inputs))

        return outputs

    def forward_output_size(self, input_size):
        output_shapes = {}

        for layer in self.to_merge:
            output_shapes.update(layer.forward_output_size(input_size))

        return output_shapes


def merge_tabular(self, other, **kwargs):
    return MergeTabular(self, other)


TabularModule.__add__ = merge_tabular
TabularModule.merge = merge_tabular
