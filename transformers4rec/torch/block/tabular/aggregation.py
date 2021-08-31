from functools import reduce

import torch

from ...utils.torch_utils import calculate_batch_size_from_input_size, requires_schema
from .tabular import TabularAggregation, tabular_aggregation_registry


@tabular_aggregation_registry.register("concat")
class ConcatFeatures(TabularAggregation):
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


@tabular_aggregation_registry.register("sequential_concat")
class SequentialConcatFeatures(TabularAggregation):
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


@tabular_aggregation_registry.register("stack")
class StackFeatures(TabularAggregation):
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


class ElementwiseFeatureAggregation(TabularAggregation):
    def _check_input_shapes_equal(self, inputs):
        all_input_shapes_equal = reduce((lambda a, b: a.shape == b.shape), inputs.values())
        if not all_input_shapes_equal:
            raise ValueError(
                "The shapes of all input features are not equal, which is required for element-wise"
                " aggregation: {}".format({k: v.shape for k, v in inputs.items()})
            )


@tabular_aggregation_registry.register("element-wise-sum")
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


@tabular_aggregation_registry.register("element-wise-sum-item-multi")
@requires_schema
class ElementwiseSumItemMulti(ElementwiseFeatureAggregation):
    def __init__(self, schema=None):
        super().__init__()
        self.stack = StackFeatures(axis=0)
        self.schema = schema
        self.item_id_col_name = None

    def forward(self, inputs):
        item_id_inputs = self.schema.get_item_ids_from_inputs(inputs)
        self._check_input_shapes_equal(inputs)

        other_inputs = {k: v for k, v in inputs.items() if k != self.item_id_col_name}
        other_inputs_sum = self.stack(other_inputs).sum(dim=0)
        result = item_id_inputs.multiply(other_inputs_sum)
        return result

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        last_dim = list(input_size.values())[0][-1]

        return batch_size, last_dim
