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
        last_dim = [i for i in input_size.values()][0][-1]

        return batch_size, len(input_size), last_dim


@aggregation_registry.register("element-wise-sum")
class ElementwiseSum(FeatureAggregation):
    def __init__(self):
        super().__init__()
        self.stack = StackFeatures(axis=0)

    def forward(self, inputs):
        return self.stack(inputs).sum(dim=0)
