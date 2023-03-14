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
from typing import Dict, Optional, Tuple, Union

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


Batch = Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]


def pad_batch(batch: Batch, padding_lengths: Dict[str, int]) -> Batch:
    """Pad list features in a batch to padding length specified

    Parameters
    ----------
    X : Batch
        dictionary of tensors in batch
    padding_lengths : Dict[str, int]
        dictionary mapping list column name to padding length

    Returns
    -------
    Batch
        Batch with padded list features

    Raises
    ------
    ValueError
        If ragged column found with no padding length provided
    """
    if batch is None or not isinstance(batch, dict):
        return batch

    batch_padded = {}
    for k, values in batch.items():
        if k.endswith("__values"):
            col_name = k[:-8]
            offsets = batch[f"{col_name}__offsets"]
            padding_length = padding_lengths.get(col_name)
            if padding_length:
                padded_values = _pad_ragged_tensor(values, offsets, padding_length)
                batch_padded[col_name] = padded_values
            else:
                # Note: This exception can be removed if the model is
                # updated to support __values / __offsets inputs
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
                col_name = k
                values, offsets = values
                padded_values = _pad_ragged_tensor(values, offsets, padding_length)
                batch_padded[col_name] = padded_values
            else:
                raise ValueError(
                    f"Found ragged column '{col_name}' with unspecified padding length. "
                    "Please provide a padding length for this feature "
                    "to be converted to a dense tensor. "
                )
        else:
            padding_length = padding_lengths.get(k)
            batch_padded[k] = _pad_dense_tensor(values, padding_length)

    return batch_padded
