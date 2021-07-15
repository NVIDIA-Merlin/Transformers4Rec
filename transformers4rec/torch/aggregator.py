import torch

from .typing import TabularData
from .utils.torch_utils import calculate_batch_size_from_input_size
from ..utils.registry import Registry

aggregators = Registry.class_registry("torch.aggregators")


class FeatureAggregator(torch.nn.Module):
    def forward(self, inputs: TabularData) -> torch.tensor:
        return super(FeatureAggregator, self).forward(inputs)


@aggregators.register("concat")
class ConcatFeatures(FeatureAggregator):
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


@aggregators.register("stack")
class StackFeatures(FeatureAggregator):
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
