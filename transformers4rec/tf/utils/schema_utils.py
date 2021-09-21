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

import tensorflow as tf

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

        for feature in schema.column_schemas:
            is_list_feature = has_field(feature, "value_count")
            is_int_feature = has_field(feature, "int_domain")
            is_embedding = feature.shape.dim[0].size > 1 if has_field(feature, "shape") else False

            shape = [d.size for d in feature.shape.dim] if has_field(feature, "shape") else (1,)

            if is_int_feature:
                max_num = feature.int_domain.max
                dtype = tf.int32
                if is_list_feature:
                    list_length = session_length or feature.value_count.max
                    row = tf.random.uniform((list_length,), minval=1, maxval=max_num, dtype=dtype)
                else:
                    row = tf.random.uniform(tuple(shape), minval=1, maxval=max_num, dtype=dtype)
            else:
                if is_list_feature:
                    list_length = session_length or feature.value_count.max
                    row = tf.random.uniform((list_length,))
                else:
                    row = tf.random.uniform(tuple(shape))

            if is_list_feature:
                row = (row, [len(row)])  # type: ignore

            if feature.name in data:
                if is_list_feature:
                    data[feature.name] = (
                        tf.concat((data[feature.name][0], row[0]), axis=0),
                        data[feature.name][1] + row[1],
                    )
                elif is_embedding:
                    f = data[feature.name]
                    if isinstance(f, list):
                        f.append(row)
                    else:
                        data[feature.name] = [f, row]
                    if i == num_rows - 1:
                        data[feature.name] = tf.stack(data[feature.name], axis=0)
                else:
                    data[feature.name] = tf.concat((data[feature.name], row), axis=0)
            else:
                data[feature.name] = row

    outputs: TabularData = {}
    for key, val in data.items():
        if isinstance(val, tuple):
            offsets = [0]
            for length in val[1][:-1]:
                offsets.append(offsets[-1] + length)
            vals = (val[0], tf.expand_dims(tf.concat(offsets, axis=0), 1))
            values, offsets, diff_offsets, num_rows = _pull_values_offsets(vals)
            indices = _get_indices(offsets, diff_offsets)
            seq_limit = max_session_length or val[1][0]
            outputs[key] = _get_sparse_tensor(values, indices, num_rows, seq_limit)
        else:
            outputs[key] = data[key]

    return outputs


def _pull_values_offsets(values_offset):
    """
    values_offset is either a tuple (values, offsets) or just values.
    Values is a tensor.
    This method is used to turn a tensor into its sparse representation
    """
    # pull_values_offsets, return values offsets diff_offsets
    if isinstance(values_offset, tuple):
        values = tf.reshape(values_offset[0], [-1])
        offsets = tf.reshape(values_offset[1], [-1])
    else:
        values = tf.reshape(values_offset, [-1])
        offsets = tf.range(tf.shape(values)[0], dtype=tf.int64)

    num_rows = len(offsets)
    offsets = tf.concat([offsets, [len(values)]], axis=0)
    diff_offsets = offsets[1:] - offsets[:-1]

    return values, offsets, diff_offsets, num_rows


def _get_indices(offsets, diff_offsets):
    # Building the indices to reconstruct the sparse tensors
    row_ids = tf.range(len(offsets) - 1, dtype=tf.int64)

    row_ids_repeated = tf.repeat(row_ids, diff_offsets)
    row_offset_repeated = tf.cast(tf.repeat(offsets[:-1], diff_offsets), tf.int64)
    col_ids = tf.range(len(row_offset_repeated), dtype=tf.int64) - row_offset_repeated
    indices = tf.concat(
        values=[tf.expand_dims(row_ids_repeated, -1), tf.expand_dims(col_ids, -1)], axis=1
    )
    return indices


def _get_sparse_tensor(values, indices, num_rows, seq_limit):
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=[num_rows, seq_limit]
    )

    return tf.sparse.to_dense(sparse_tensor)
