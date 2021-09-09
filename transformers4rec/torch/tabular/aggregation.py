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

from functools import reduce

import torch

from merlin_standard_lib import Schema, Tag

from ...config.schema import requires_schema
from ..typing import TabularData
from ..utils.torch_utils import calculate_batch_size_from_input_size
from .tabular import TabularAggregation, tabular_aggregation_registry


@tabular_aggregation_registry.register("concat")
class ConcatFeatures(TabularAggregation):
    """Aggregation by concatenating all values in the input dictionary in the given dimension.

    Parameters
    ----------
    axis: int, default=-1
        Axis to use for the concatenation operation.
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs: TabularData) -> torch.Tensor:
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])

        return torch.cat(tensors, dim=self.axis)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)

        return batch_size, sum([i[1] for i in input_size.values()])


@tabular_aggregation_registry.register("sequential-concat")
@requires_schema
class SequentialConcatFeatures(TabularAggregation):
    """Aggregation by stacking all values in TabularData, all non-sequential values will be
    converted to a sequence.

    The output of this concatenation will have 3 dimensions.
    """

    def __init__(self, schema: Schema = None):
        super().__init__()
        self.schema = schema

    def forward(
        self,
        inputs: TabularData,
    ) -> torch.Tensor:
        continuous_features_names = self.schema.select_by_tag(Tag.CONTINUOUS).column_names
        inputs_sizes = {k: v.shape for k, v in inputs.items()}
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(
            inputs_sizes, continuous_features_names
        )

        if len(seq_features_shapes) > 0:
            for fname in inputs:
                if fname in continuous_features_names:
                    # For continuous features, include and additional dim in the end to match
                    # the number of dim of embedding features
                    inputs[fname] = inputs[fname].unsqueeze(dim=-1)

            non_seq_features = set(inputs.keys()).difference(set(seq_features_shapes.keys()))
            for fname in non_seq_features:
                # Including the 2nd dim and repeating for the sequence length
                inputs[fname] = inputs[fname].unsqueeze(dim=1).repeat(1, sequence_length, 1)

        output_sizes = {k: v.shape for k, v in inputs.items()}
        self._check_concat_shapes(output_sizes)

        tensors = []
        for name in sorted(inputs.keys()):
            val = inputs[name]
            tensors.append(val)

        return torch.cat(tensors, dim=-1)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)

        continuous_features_names = self.schema.select_by_tag(Tag.CONTINUOUS).column_names

        seq_features_shapes, sequence_length = self._get_seq_features_shapes(
            input_size, continuous_features_names
        )

        if len(seq_features_shapes) > 0:
            last_dims = []
            for fname, fvalue in input_size.items():
                if fname in continuous_features_names:
                    last_dims.append(1)
                else:
                    last_dims.append(fvalue[-1])

            concat_last_dims = sum(last_dims)

            return (
                batch_size,
                sequence_length,
                concat_last_dims,
            )

        else:
            self._check_concat_shapes(input_size)
            concat_last_dims = sum([v[-1] for v in input_size.values()])
            return (batch_size, concat_last_dims)

    def _get_seq_features_shapes(self, inputs_sizes, continuous_features_names):

        seq_features_shapes = dict()
        for fname, fshape in inputs_sizes.items():
            # Saves the shapes of sequential features
            if (fname in continuous_features_names and len(fshape) >= 2) or (
                fname not in continuous_features_names and len(fshape) >= 3
            ):
                seq_features_shapes[fname] = tuple(fshape[:2])

        sequence_length = 0
        if len(seq_features_shapes) > 0:
            # all_shapes_equal = reduce((lambda a, b: a == b), seq_features_shapes.values())
            if len(set(seq_features_shapes.values())) > 1:
                raise ValueError(
                    "All sequential features must share the same shape in the first two dims "
                    "(batch_size, seq_length): {}".format(seq_features_shapes)
                )

            sequence_length = list(seq_features_shapes.values())[0][1]

        return seq_features_shapes, sequence_length

    def _check_concat_shapes(self, input_sizes):
        if len(set(list([v[:-1] for v in input_sizes.values()]))) > 1:
            raise Exception(
                "All features dimensions except the last one must match: {}".format(input_sizes)
            )


@tabular_aggregation_registry.register("stack")
class StackFeatures(TabularAggregation):
    """Aggregation by stacking all values in input dictionary in the given dimension.

    Parameters
    ----------
    axis: int, default=-1
        Axis to use for the stacking operation.
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs: TabularData) -> torch.Tensor:
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
    """Aggregation by first stacking all values in TabularData in the first dimension, and then
    summing the result."""

    def __init__(self):
        super().__init__()
        self.stack = StackFeatures(axis=0)

    def forward(self, inputs: TabularData) -> torch.Tensor:
        self._check_input_shapes_equal(inputs)
        return self.stack(inputs).sum(dim=0)

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        last_dim = list(input_size.values())[0][-1]

        return batch_size, last_dim


@tabular_aggregation_registry.register("element-wise-sum-item-multi")
@requires_schema
class ElementwiseSumItemMulti(ElementwiseFeatureAggregation):
    """Aggregation by applying the `ElementwiseSum` aggregation to all features except the item-id,
    and then multiplying this with the item-ids.

    Parameters
    ----------
    schema: DatasetSchema
    """

    def __init__(self, schema: Schema = None):
        super().__init__()
        self.stack = StackFeatures(axis=0)
        self.schema = schema
        self.item_id_col_name = None

    def forward(self, inputs: TabularData) -> torch.Tensor:
        item_id_inputs = self.get_item_ids_from_inputs(inputs)
        self._check_input_shapes_equal(inputs)

        other_inputs = {k: v for k, v in inputs.items() if k != self.schema.item_id_column_name}
        other_inputs_sum = self.stack(other_inputs).sum(dim=0)
        result = item_id_inputs.multiply(other_inputs_sum)
        return result

    def forward_output_size(self, input_size):
        batch_size = calculate_batch_size_from_input_size(input_size)
        last_dim = list(input_size.values())[0][-1]

        return batch_size, last_dim
