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

import random
from typing import Any, Dict, Optional

import torch

from merlin_standard_lib import Schema
from merlin_standard_lib.utils.proto_utils import has_field

from ..typing import TabularData


def random_data_from_schema(
    schema: Schema,
    num_rows: int,
    max_session_length: Optional[int] = None,
    min_session_length: int = 5,
) -> TabularData:
    data: Dict[str, Any] = {}

    for i in range(num_rows):
        session_length = None
        if max_session_length:
            session_length = random.randint(min_session_length, max_session_length)

        for feature in schema.feature:
            is_list_feature = has_field(feature, "value_count")
            is_int_feature = has_field(feature, "int_domain")
            is_embedding = feature.shape.dim[0].size > 1 if has_field(feature, "shape") else False

            shape = [d.size for d in feature.shape.dim] if has_field(feature, "shape") else (1,)

            if is_int_feature:
                max_num = feature.int_domain.max
                if is_list_feature:
                    list_length = session_length or feature.value_count.max
                    row = torch.randint(1, max_num, (list_length,))

                else:
                    row = torch.randint(1, max_num, tuple(shape))
            else:
                if is_list_feature:
                    list_length = session_length or feature.value_count.max
                    row = torch.rand((list_length,))
                else:
                    row = torch.rand(tuple(shape))

            if is_list_feature:
                row = (row, [len(row)])  # type: ignore

            if feature.name in data:
                if is_list_feature:
                    data[feature.name] = (
                        torch.cat((data[feature.name][0], row[0])),
                        data[feature.name][1] + row[1],
                    )
                elif is_embedding:
                    f = data[feature.name]
                    if isinstance(f, list):
                        f.append(row)
                    else:
                        data[feature.name] = [f, row]
                    if i == num_rows - 1:
                        data[feature.name] = torch.stack(data[feature.name], dim=0)
                else:
                    data[feature.name] = torch.cat((data[feature.name], row))
            else:
                data[feature.name] = row

    outputs: TabularData = {}
    for key, val in data.items():
        if isinstance(val, tuple):
            offsets = [0]
            for length in val[1][:-1]:
                offsets.append(offsets[-1] + length)
            vals = (val[0], torch.tensor(offsets).unsqueeze(dim=1))
            values, offsets, diff_offsets, num_rows = _pull_values_offsets(vals)
            indices = _get_indices(offsets, diff_offsets)
            seq_limit = max_session_length or val[1][0]
            outputs[key] = _get_sparse_tensor(values, indices, num_rows, seq_limit)
        else:
            outputs[key] = data[key]

    return outputs


def _pull_values_offsets(values_offset):
    # pull_values_offsets, return values offsets diff_offsets
    if isinstance(values_offset, tuple):
        values = values_offset[0].flatten()
        offsets = values_offset[1].flatten()
    else:
        values = values_offset.flatten()
        offsets = torch.arange(values.size()[0])
    num_rows = len(offsets)
    offsets = torch.cat([offsets, torch.tensor([len(values)])])
    diff_offsets = offsets[1:] - offsets[:-1]
    return values, offsets, diff_offsets, num_rows


def _get_indices(offsets, diff_offsets):
    row_ids = torch.arange(len(offsets) - 1)
    row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
    col_ids = torch.arange(len(row_offset_repeated)) - row_offset_repeated
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices


def _get_sparse_tensor(values, indices, num_rows, seq_limit):
    sparse_tensor = torch.sparse_coo_tensor(indices.T, values, torch.Size([num_rows, seq_limit]))

    return sparse_tensor.to_dense()
