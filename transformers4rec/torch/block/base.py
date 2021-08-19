import abc
import inspect
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch.nn import Module

from transformers4rec.torch.typing import MaskSequence
from transformers4rec.torch.utils.torch_utils import calculate_batch_size_from_input_size

from ..head import Head
from ..tabular import AsTabular, FilterFeatures, TabularMixin, TabularModule, merge_tabular


class BlockMixin:
    def to_model(
        self,
        head: Head,
        masking: Optional[Union[MaskSequence, str]] = None,
        optimizer=torch.optim.Adam,
        block_output_size=None,
        **kwargs
    ):
        from .with_head import BlockWithHead

        if not block_output_size:
            if getattr(self, "input_size", None) and getattr(self, "forward_output_size", None):
                block_output_size = self.forward_output_size(self.input_size)
        if block_output_size:
            self.output_size = block_output_size

        out = BlockWithHead(self, head, masking=masking, optimizer=optimizer, **kwargs)

        if next(self.parameters()).is_cuda:
            out.to("cuda")

        return out


class TabularBlock(TabularModule, BlockMixin):
    pass


class Block(BlockMixin, torch.nn.Module):
    def __init__(self, module: torch.nn.Module, output_size):
        super().__init__()
        self.module = module
        self._output_size = output_size

    def forward(self, inputs):
        return self.module(inputs)

    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> AsTabular(name)

    def output_size(self):
        if self._output_size[0] is None:
            batch_size = calculate_batch_size_from_input_size(self.input_shape)

            return batch_size + self._output_size

        return self._output_size

    def forward_output_size(self, input_size):
        if self._output_size[0] is None:
            batch_size = calculate_batch_size_from_input_size(input_size)

            return [batch_size] + self._output_size[1:]

        return self._output_size

    def build(self, input_shape):
        self.input_shape = input_shape
        self._built = True

        return self


class SequentialBlock(TabularMixin, BlockMixin, torch.nn.Sequential):
    def __init__(self, *args, output_size=None):
        super().__init__()
        self._static_output_size = output_size
        self.input_size = None
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            last = None
            for idx, key, module in enumerate(args[0].items()):
                self.add_module_and_maybe_build(key, module, last, idx)
                last = module
        else:
            last = None
            for idx, module in enumerate(args):
                last = self.add_module_and_maybe_build(str(idx), module, last, idx)

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

        return self >> AsTabular(name)

    def __add__(self, other):
        return merge_tabular(self, other)

    def forward_output_size(self, input_size):
        if self._static_output_size:
            return self._static_output_size

        x = input_size
        for module in list(self):
            if getattr(module, "forward_output_size", None):
                x = module.forward_output_size(x)
            else:
                # TODO log warning here
                return None

        return x

    # TODO: seems like this gets overriden on line 28?
    # pylint: disable=method-hidden
    def output_size(self):
        if not getattr(self, "input_size", None):
            # TODO: log warning here
            pass
        return self.forward_output_size(self.input_size)

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
    def build(self, input_shape) -> Union[TabularBlock, Block, SequentialBlock]:
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


BlockType = Union[torch.nn.Module, Block]


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
        need_moving_to_gpu = need_moving_to_gpu or _check_gpu(self)
    if isinstance(other, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or _check_gpu(other)

    out = SequentialBlock(*sequential)
    if getattr(left_side[-1], "input_size", None) and left_side[-1].input_size:
        out.input_size = left_side[-1].input_size

    if need_moving_to_gpu:
        out.to("cuda")

    return out


def _check_gpu(module):
    try:
        return next(module.parameters()).is_cuda
    except StopIteration:
        return False
