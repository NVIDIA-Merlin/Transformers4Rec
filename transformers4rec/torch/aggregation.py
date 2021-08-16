import abc
from functools import reduce

import torch

from ..utils.registry import Registry
from .typing import TabularData
from .utils.torch_utils import calculate_batch_size_from_input_size

aggregation_registry: Registry = Registry.class_registry("torch.aggregation_registry")


class FeatureAggregation(torch.nn.Module):
    def __rrshift__(self, other):
        from .block.base import right_shift_block

        return right_shift_block(self, other)

    def forward(self, inputs: TabularData) -> torch.tensor:
        return super(FeatureAggregation, self).forward(inputs)

    @abc.abstractmethod
    def forward_output_size(self, input_size):
        raise NotImplementedError

    def build(self, input_size, device=None):
        if device:
            self.to(device)
        self.input_size = input_size

        return self

    def output_size(self):
        if not self.input_size:
            # TODO: log warning here
            pass

        return self.forward_output_size(self.input_size)

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, value):
        self._schema = value


@aggregation_registry.register("concat")
class ConcatFeatures(FeatureAggregation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])

        return torch.cat(tensors, dim=self.axis)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)

        return batch_size, sum([i[1] for i in input_size.values()])


@aggregation_registry.register("sequential_concat")
class SequentialConcatFeatures(FeatureAggregation):
    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            val = inputs[name]
            if val.ndim == 2:
                val = val.unsqueeze(dim=-1)
            tensors.append(val)

        return torch.cat(tensors, dim=-1)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        converted_input_size = {}
        for key, val in input_size.items():
            if len(val) == 2:
                converted_input_size[key] = val + (1,)
            else:
                converted_input_size[key] = val

        return (
            batch_size,
            list(input_size.values())[0][1],
            sum([i[-1] for i in converted_input_size.values()]),
        )


@aggregation_registry.register("stack")
class StackFeatures(FeatureAggregation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])

        return torch.stack(tensors, dim=self.axis)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        last_dim = list(input_size.values())[0][-1]

        return batch_size, len(input_size), last_dim


class ElementwiseFeatureAggregation(FeatureAggregation):
    def _check_input_shapes_equal(self, inputs):
        all_input_shapes_equal = reduce((lambda a, b: a.shape == b.shape), inputs.values())
        if not all_input_shapes_equal:
            raise ValueError(
                "The shapes of all input features are not equal, which is required for element-wise"
                " aggregation: {}".format({k: v.shape for k, v in inputs.items()})
            )


@aggregation_registry.register("element-wise-sum")
class ElementwiseSum(ElementwiseFeatureAggregation):
    def __init__(self):
        super().__init__()
        self.stack = StackFeatures(axis=0)

    def forward(self, inputs):
        self._check_input_shapes_equal(inputs)
        return self.stack(inputs).sum(dim=0)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        last_dim = list(input_size.values())[0][-1]

        return batch_size, last_dim


@aggregation_registry.register("element-wise-sum-item-multi")
class ElementwiseSumItemMulti(ElementwiseFeatureAggregation):
    def __init__(self, item_id=None):
        super().__init__()
        self.stack = StackFeatures(axis=0)
        self.item_id = item_id

    def _check_input_id(self, inputs):
        if not self.item_id:
            raise ValueError("The `item_id` is necessary to apply this aggregation.")

        if not inputs or self.item_id not in inputs:
            raise ValueError(f"{self.item_id} not found in inputs")

    def forward(self, inputs):
        self._check_input_id(inputs)
        self._check_input_shapes_equal(inputs)

        item_id_inputs = inputs[self.item_id]
        other_inputs = {k: v for k, v in inputs.items() if k != self.item_id}
        other_inputs_sum = self.stack(other_inputs).sum(dim=0)
        result = item_id_inputs.multiply(other_inputs_sum)
        return result

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        last_dim = list(input_size.values())[0][-1]

        return batch_size, last_dim
