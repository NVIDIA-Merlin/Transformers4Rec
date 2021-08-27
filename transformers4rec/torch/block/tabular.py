import abc
import logging
from functools import reduce

import torch

from ..tabular import TabularModule
from ..typing import TabularData
from .base import BlockBase, right_shift_block

LOG = logging.getLogger("transformers4rec")


class TabularBlock(BlockBase, TabularModule, abc.ABC):
    def to_module(self, shape_or_module, device=None):
        shape = shape_or_module
        if isinstance(shape_or_module, torch.nn.Module):
            shape = getattr(shape_or_module, "output_size", None)
            if shape:
                shape = shape()

        return self.build(shape, device=device)

    def output_size(self, input_size=None):
        output_size = self._check_aggregation_output_size(super().output_size(input_size))

        return output_size

    def _check_aggregation_output_size(self, input_size):
        output_size = input_size
        if isinstance(input_size, dict) and self.aggregation:
            output_size = self.aggregation.forward_output_size(input_size)

        return output_size

    def __rrshift__(self, other):
        return right_shift_block(self, other)


class MergeTabular(TabularBlock):
    def __init__(self, *to_merge, aggregation=None, augmentation=None):
        super().__init__(aggregation=aggregation, augmentation=augmentation)
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
