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
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _pad_dense_tensor(t: torch.Tensor, length: Optional[int]) -> torch.Tensor:
    if length and len(t.shape) == 2:
        pad_diff = length - t.shape[1]
        return F.pad(input=t, pad=(0, pad_diff, 0, 0))
    return t


def _squeeze(tensor):
    if len(tensor.shape) == 2:
        return tensor.squeeze(1)
    return tensor


def _get_indices(offsets, diff_offsets):
    row_ids = torch.arange(len(offsets) - 1, device=offsets.device)
    row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
    col_ids = torch.arange(len(row_offset_repeated), device=offsets.device) - row_offset_repeated
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices


def _pad_ragged_tensor(values, offsets, padding_length):
    values = _squeeze(values)
    offsets = _squeeze(offsets)
    num_rows = len(offsets) - 1
    diff_offsets = offsets[1:] - offsets[:-1]
    indices = _get_indices(offsets, diff_offsets)
    sparse_tensor = torch.sparse_coo_tensor(
        indices.T, values, torch.Size([num_rows, padding_length]), device=values.device
    )
    return sparse_tensor.to_dense()


def _pad_batch(X, padding_lengths, ragged_pad_fn):
    if X is None or not isinstance(X, dict):
        return X

    X_padded = {}
    for k, values in X.items():
        if k.endswith("__values"):
            col_name = k[:-8]
            offsets = X[f"{col_name}__offsets"]
            padding_length = padding_lengths.get(col_name)
            if padding_length:
                padded_values = ragged_pad_fn(values, offsets, padding_length)
                X_padded[col_name] = padded_values
            else:
                raise ValueError(
                    f"Found ragged column '{col_name}' with unspecified padding length. "
                    "Please provide a padding length for this feature "
                    "to be converted to a dense tensor. "
                )
        elif k.endswith("__offsets"):
            continue
        elif isinstance(values, tuple):
            padding_length = padding_lengths.get(k)
            if padding_length:
                values, offsets = values
                padded_values = ragged_pad_fn(values, offsets, padding_length)
                X_padded[k] = padded_values
            else:
                X_padded[k] = values
        else:
            padding_length = padding_lengths.get(k)
            X_padded[k] = _pad_dense_tensor(values, padding_length)

    return X_padded


def get_pad_fn(padding_lengths: Dict[str, int]):
    def pad_fn(x, y):
        new_x = _pad_batch(x, padding_lengths, _pad_ragged_tensor)
        new_y = _pad_batch(y, padding_lengths, _pad_ragged_tensor)
        return new_x, new_y

    return pad_fn
