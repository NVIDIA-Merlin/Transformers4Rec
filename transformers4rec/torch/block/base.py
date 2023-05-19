#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import abc
import inspect
import logging
from collections import OrderedDict
from typing import List, Optional, Union

import torch
from merlin.models.utils.misc_utils import filter_kwargs
from torch.nn import Module

from ..utils import torch_utils

LOG = logging.getLogger("transformers4rec")


class BlockBase(torch_utils.OutputSizeMixin, torch.nn.Module, metaclass=abc.ABCMeta):
    """A subclass of PyTorch's torch.nn.Module, providing additional functionality
    for dealing with automatic setting of input/output dimensions of neural networks layers.
    Specifically, It implements the 'OutputSizeMixin' for managing output sizes.
    """

    def to_model(self, prediction_task_or_head, inputs=None, **kwargs):
        """Converts the BlockBase instance into a T4Rec model by attaching it to
        attaching a 'Head' or 'PredictionTask'.

        Parameters
        ----------
        prediction_task_or_head : Union[PredictionTask, Head]
            A PredictionTask or Head instance to attach to this block.
        inputs :InputBlock, optional
            The input block representing input features.
            By default None

        Raises
        ------
        ValueError
            If prediction_task_or_head is neither a Head nor a PredictionTask.
        """
        from ..model.base import Head, Model, PredictionTask

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
        """Converts the output of the block into a dictionary, keyed by the
        provided name

        Parameters
        ----------
        name : str, optional
            The output name, if not provided, uses the name of the block class.
            by default None
        """
        from ..tabular.base import AsTabular

        if not name:
            name = self.name

        return SequentialBlock(self, AsTabular(name))


class Block(BlockBase):
    """Wraps a PyTorch module, allowing it to be used as a block in a T4Rec model.
    It carries the module and its expected output size.

    Parameters
    ----------
    module: torch.nn.Module
        The PyTorch module to be wrapped in this block.
    output_size: Union[List[int], torch.Size]
        The expected output size of the module.

    """

    def __init__(self, module: torch.nn.Module, output_size: Union[List[int], torch.Size]):
        super().__init__()
        self.module = module
        self._output_size = output_size

    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)

    def forward_output_size(self, input_size: Union[List[int], torch.Size]):
        """
        Calculates the output size of the tensor(s) returned by the forward pass,
        given the input size.

        Parameters
        ----------
        input_size:  Union[List[int], torch.Size]
            The size of the input tensor(s) to the module.

        Returns
        -------
        Union[List[int], torch.Size]
            The size of the output from the module.
        """
        if self._output_size[0] is None:
            batch_size = torch_utils.calculate_batch_size_from_input_size(input_size)

            return [batch_size] + self._output_size[1:]

        return self._output_size


class SequentialBlock(BlockBase, torch.nn.Sequential):
    """Extends the module torch.nn.Sequential. It's used for creating
    a sequence of layers or blocks in a T4Rec model. The modules
    will be applied to inputs in the order they are passed in the constructor.

    Parameters
    ----------
    *args: torch.nn.Module
        The list of PyTorch modules.
    output_size : Union[List[int], torch.Size], optional
        The expected output size from the last layer in the sequential block
        By default None

    """

    def __init__(self, *args, output_size: Union[List[int], torch.Size] = None):
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

    def add_module(self, name: str, module: Optional[Module]) -> None:
        """
        Adds a PyTorch module to the sequential block. If a list of strings is provided,
        a `FilterFeatures` block gets added to the sequential block.

        Parameters
        ----------
        name : str
            The name of the child module. The child module can be accessed
            from this module using the given name.
        module : Optional[Union[List[str], Module]]
            The child module to be added to the module.
        """
        from ..tabular.base import FilterFeatures

        if isinstance(module, list):
            module = FilterFeatures(module)
        super().add_module(name, module)

    def add_module_and_maybe_build(self, name: str, module, parent, idx) -> torch.nn.Module:
        """Checks if a module needs to be built and adds it to the sequential block.

        Parameters
        ----------
        name : str
            The name of the child module.
        module : torch.nn.Module
            The child module to be added to the sequential block.
        parent : torch.nn.Module
            The parent module.
        idx : int
            The index of the current module in the sequential block.
        """
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

    def forward(self, input, training=False, testing=False, **kwargs):
        """Applies the module's layers sequentially to the input block.

        Parameters
        ----------
        input : tensor
            The input to the block.
        training : bool, optional
            Whether the block is in training mode. The default is False.
        testing : bool, optional
            Whether the block is in testing mode. The default is False.

        """
        for i, module in enumerate(self):
            if i == len(self) - 1:
                filtered_kwargs = filter_kwargs(kwargs, module, cascade_kwargs_if_possible=True)
                input = module(input, **filtered_kwargs)

            elif "training" in inspect.signature(module.forward).parameters:
                if "testing" in inspect.signature(module.forward).parameters:
                    input = module(input, training=training, testing=testing)
                else:
                    input = module(input, training=training)
            else:
                input = module(input)

        return input

    def build(self, input_size, schema=None, **kwargs):
        """Builds the layers of the sequential block given the input size.

        Parameters
        ----------
        input_size : Union[List[int], torch.Size]
            The size of the input tensor(s).
        schema : Schema, optional
            The schema of the inputs features, by default None

        Returns
        -------
        SequentialBlock
            The built sequential block.
        """
        output_size = input_size
        for module in self:
            if not hasattr(module, "build"):
                break
            module.build(output_size, schema=schema)
            output_size = module.output_size()

        return super(SequentialBlock, self).build(input_size, schema=None, **kwargs)

    def as_tabular(self, name=None):
        """Converts the output of the block into a dictionary, keyed by the
        provided name

        Parameters
        ----------
        name : str, optional
            The output name, if not provided, uses the name of the block class.
            by default None
        """
        from transformers4rec.torch import AsTabular

        if not name:
            name = self.name

        return SequentialBlock(self, AsTabular(name))

    def __add__(self, other):
        from ..tabular.base import merge_tabular

        return merge_tabular(self, other)

    def forward_output_size(self, input_size):
        """
        Calculates the output size of the tensor(s) returned by the forward pass,
        given the input size.

        Parameters
        ----------
        input_size:  Union[List[int], torch.Size]
            The size of the input tensor(s) to the module.

        Returns
        -------
        Union[List[int], torch.Size]
            The size of the output from the module.
        """
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
    """Builds a SequentialBlock from a list of PyTorch modules.

    Parameters
    ----------
    *modules : List[torch.nn.Module]
        List containing PyTorch modules.

    Returns
    -------
        A SequentialBlock instance created from the provided modules.
    """
    return list(SequentialBlock(*modules))


class BuildableBlock(abc.ABC):
    """
    Abstract base class for buildable blocks.
    Subclasses of BuildableBlock must implement the `build` method
    """

    @abc.abstractmethod
    def build(self, input_size) -> BlockBase:
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
    from ..tabular.base import FilterFeatures

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


BlockType = Union[BlockBase, BuildableBlock]
BlockOrModule = Union[BlockBase, BuildableBlock, torch.nn.Module]
