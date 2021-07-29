import abc
from typing import Union

import torch

from ..head import Head
from ..tabular import AsTabular, FilterFeatures, TabularMixin, TabularModule, merge_tabular


class BlockMixin:
    def to_model(self, head: Head, optimizer=torch.optim.Adam, block_output_size=None, **kwargs):
        from .with_head import BlockWithHead

        if not block_output_size:
            if getattr(self, "input_size", None) and getattr(self, "forward_output_size", None):
                block_output_size = self.forward_output_size(self.input_size)
        if block_output_size:
            self.output_size = block_output_size

        out = BlockWithHead(self, head, optimizer=optimizer, **kwargs)

        if next(self.parameters()).is_cuda:
            out.to("cuda")

        return out


class TabularBlock(TabularModule, BlockMixin):
    pass


class Block(BlockMixin, torch.nn.Module):
    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> AsTabular(name)


class SequentialBlock(TabularMixin, BlockMixin, torch.nn.Sequential):
    def __rrshift__(self, other):
        return right_shift_block(self, other)

    def __rshift__(self, other):
        return right_shift_block(other, self)

    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> AsTabular(name)

    def __add__(self, other):
        return merge_tabular(self, other)

    def forward_output_size(self, input_size):
        x = input_size
        for module in list(self):
            if getattr(module, "forward_output_size", None):
                x = module.forward_output_size(x)
            else:
                # TODO log warning here
                return None

        return x

    def output_size(self):
        if not getattr(self, "input_size", None):
            # TODO: log warning here
            pass

        return self.forward_output_size(self.input_size)


class BuildableBlock(abc.ABC):
    @abc.abstractmethod
    def build(self, input_shape) -> Union[TabularBlock, Block, SequentialBlock]:
        raise NotImplementedError

    def __rrshift__(self, other):
        return right_shift_block(self, other)


BlockType = Union[torch.nn.Module, Block]


def right_shift_block(self, other):
    if isinstance(other, list):
        left_side = [FilterFeatures(other)]
    else:
        left_side = list(other) if isinstance(other, SequentialBlock) else [other]
    right_side = list(self) if isinstance(self, SequentialBlock) else [self]

    # Maybe build right-side
    if getattr(left_side[-1], "output_size", None) and left_side[-1].output_size():
        _right_side = []
        x = left_side[-1].output_size()
        for module in right_side:
            if getattr(module, "build", None):
                build = module.build(x)
                if build:
                    module = build
                x = module.output_size()
            _right_side.append(module)
        right_side = _right_side

    sequential = left_side + right_side

    need_moving_to_gpu = False
    if isinstance(self, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or _check_gpu(self)
    if isinstance(other, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or _check_gpu(other)

    out = SequentialBlock(*sequential)

    if need_moving_to_gpu:
        out.to("cuda")

    return out


def _check_gpu(module):
    try:
        return next(module.parameters()).is_cuda
    except StopIteration:
        return False
