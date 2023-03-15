#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
from itertools import accumulate

import torch

from transformers4rec.torch.utils.padding import pad_batch


def _get_values_offsets(data):
    values = []
    row_lengths = []
    for row in data:
        row_lengths.append(len(row))
        values += row
    offsets = [0] + list(accumulate(row_lengths))
    return torch.tensor(values), torch.tensor(offsets)


def test_pad_values_offsets_tuple():
    data = [[1, 2], [], [3, 4, 5]]
    values, offsets = _get_values_offsets(data)

    x = {"a": (values, offsets)}

    padded_x = pad_batch(x, {"a": 5})
    assert torch.equal(
        padded_x["a"],
        torch.tensor(
            [
                [1, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [3, 4, 5, 0, 0],
            ]
        ),
    )


def test_pad_values_offsets_dict():
    data = [[1, 2], [], [3, 4, 5]]
    values, offsets = _get_values_offsets(data)

    x = {"a__values": values, "a__offsets": offsets}

    padded_x = pad_batch(x, {"a": 7})
    assert torch.equal(
        padded_x["a"],
        torch.tensor(
            [
                [1, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0],
            ]
        ),
    )


def test_pad_values_dense():
    data = [[1, 2], [], [3, 4, 5]]
    values, offsets = _get_values_offsets(data)

    x = {"a__values": values, "a__offsets": offsets, "b": torch.tensor([[3, 6], [4, 1], [8, 4]])}

    padded_x = pad_batch(x, {"a": 7, "b": 3})
    assert torch.equal(
        padded_x["a"],
        torch.tensor(
            [
                [1, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        padded_x["b"],
        torch.tensor(
            [
                [3, 6, 0],
                [4, 1, 0],
                [8, 4, 0],
            ]
        ),
    )
