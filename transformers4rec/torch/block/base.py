import abc
import inspect
import logging
from abc import ABC, ABCMeta
from collections import OrderedDict
from functools import reduce
from typing import List, Optional, Union

import torch
from torch.nn import Module

from ..tabular import FilterFeatures, TabularModule, merge_tabular
from ..typing import Head, PredictionTask, TabularData
from ..utils import torch_utils

LOG = logging.getLogger("transformers4rec")


class BlockBase(torch.nn.Module, torch_utils.OutputSizeMixin, metaclass=ABCMeta):
    def to_model(self, prediction_task_or_head: Union[PredictionTask, Head], inputs=None, **kwargs):
        from ..head import Head, PredictionTask
        from ..model import Model

        if isinstance(prediction_task_or_head, PredictionTask):
            head = prediction_task_or_head.to_head(self, inputs=inputs, **kwargs)
        elif isinstance(prediction_task_or_head, Head):
            head = prediction_task_or_head
        else:
            raise ValueError(
                "`prediction_task_or_head` needs to be a `Head` or `PredictionTask` "
                f"found: {type(prediction_task_or_head)}"
            )

        return Model(head, **kwargs)

    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return SequentialBlock(self, AsTabular(name))


class TabularBlock(BlockBase, TabularModule, ABC):
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


class Block(BlockBase):
    def __init__(self, module: torch.nn.Module, output_size: Union[List[int], torch.Size]):
        super().__init__()
        self.module = module
        self._output_size = output_size

    def forward(self, inputs):
        return self.module(inputs)

    def forward_output_size(self, input_size):
        if self._output_size[0] is None:
            batch_size = torch_utils.calculate_batch_size_from_input_size(input_size)

            return [batch_size] + self._output_size[1:]

        return self._output_size


class SequentialBlock(BlockBase, torch.nn.Sequential):
    def __init__(self, *args, output_size=None):
        from transformers4rec.torch import TabularSequenceFeatures, TransformerBlock

        if isinstance(args[0], TabularSequenceFeatures) and any(
            isinstance(arg, TransformerBlock) for arg in args
        ):
            masking = args[0].masking
            for arg in args:
                if isinstance(arg, TransformerBlock):
                    if arg.masking != masking:
                        LOG.warning(
                            "Masking is set in the input module but not in the "
                            "TransformerBlock, provide this through the masking argument"
                        )

        super().__init__()
        self._static_output_size = output_size
        self.input_size = None

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            last = None
            for idx, key, module in enumerate(args[0].items()):
                self.add_module_and_maybe_build(key, module, last, idx)
                last = module
        else:
            if len(args) == 1 and isinstance(args[0], list):
                args = args[0]
            last = None
            for idx, module in enumerate(args):
                last = self.add_module_and_maybe_build(str(idx), module, last, idx)

    @property
    def inputs(self):
        from transformers4rec.torch import TabularFeatures, TabularSequenceFeatures

        first = list(self)[0]
        if isinstance(first, (TabularSequenceFeatures, TabularFeatures)):
            return first

    def add_module(self, name: str, module: Optional[Union[Module, str]]) -> None:
        if isinstance(module, list):
            module = FilterFeatures(module)
        super().add_module(name, module)

    def add_module_and_maybe_build(self, name: str, module, parent, idx) -> torch.nn.Module:
        # Check if module needs to be built
        if getattr(parent, "output_size", None) and getattr(module, "build", None):
            module = module.build(parent.output_size())

        if idx == 0:
            self.input_size = getattr(module, "input_size", None)

        self.add_module(name, module)

        return module

    def __rrshift__(self, other):
        return right_shift_block(self, other)

    def __rshift__(self, other):
        # pylint: disable=arguments-out-of-order
        return right_shift_block(other, self)

    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return SequentialBlock(self, AsTabular(name))

    def __add__(self, other):
        return merge_tabular(self, other)

    def forward_output_size(self, input_size):
        if self._static_output_size:
            return self._static_output_size

        x = input_size
        for module in list(self):
            if getattr(module, "output_size", None):
                x = module.output_size(x)
            else:
                # TODO log warning here
                return None

        return x

    @staticmethod
    def get_children_by_class_name(parent, *class_name):
        children = []

        def add_if_class_name_matches(to_check):
            if to_check.__class__.__name__ in class_name:
                children.append(to_check)

        for child in parent:
            if getattr(child, "merge_values", None):
                for to_merge in child.merge_values:
                    add_if_class_name_matches(to_merge)

            add_if_class_name_matches(child)

        return children


def build_blocks(*modules):
    return list(SequentialBlock(*modules))


class BuildableBlock(abc.ABC):
    @abc.abstractmethod
    def build(self, input_size) -> Union[TabularBlock, Block, SequentialBlock]:
        raise NotImplementedError

    def __rrshift__(self, other):
        return right_shift_block(self, other)

    def to_module(self, shape_or_module):
        shape = shape_or_module
        if isinstance(shape_or_module, torch.nn.Module):
            shape = getattr(shape_or_module, "output_size", None)
            if shape:
                shape = shape()

        return self.build(shape)


def right_shift_block(self, other):
    if isinstance(other, list):
        left_side = [FilterFeatures(other)]
    else:
        left_side = list(other) if isinstance(other, SequentialBlock) else [other]
    right_side = list(self) if isinstance(self, SequentialBlock) else [self]

    # Maybe build right-side
    if hasattr(left_side[-1], "output_size") and left_side[-1].output_size():
        _right_side = []
        x = left_side[-1].output_size()
        for module in right_side:
            if getattr(module, "build", None):
                if "parents" in inspect.signature(module.build).parameters:
                    build = module.build(x, left_side)
                else:
                    build = module.build(x)
                if build:
                    module = build
                x = module.output_size() if hasattr(module, "output_size") else None
            _right_side.append(module)
        right_side = _right_side

    sequential = left_side + right_side

    need_moving_to_gpu = False
    if isinstance(self, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or torch_utils.check_gpu(self)
    if isinstance(other, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or torch_utils.check_gpu(other)

    out = SequentialBlock(*sequential)
    if getattr(left_side[-1], "input_size", None) and left_side[-1].input_size:
        out.input_size = left_side[-1].input_size

    if need_moving_to_gpu:
        out.to("cuda")

    return out
