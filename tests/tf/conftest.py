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

import pathlib
import random

import numpy as np
import pytest

from merlin_standard_lib import Schema
from merlin_standard_lib.utils.proto_utils import has_field, proto_text_to_better_proto

tf = pytest.importorskip("tensorflow")
tf4rec = pytest.importorskip("transformers4rec.tf")

NUM_EXAMPLES = 1000
MAX_CARDINALITY = 100
ASSETS_DIR = pathlib.Path(__file__).parent.parent / "assets"


@pytest.fixture
def tf_con_features():
    features = {}
    keys = "abcdef"

    for key in keys:
        features[key] = tf.random.uniform((NUM_EXAMPLES, 1))

    return features


@pytest.fixture
def tf_cat_features():
    features = {}
    keys = [f"cat_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = tf.random.uniform((NUM_EXAMPLES, 1), maxval=MAX_CARDINALITY, dtype=tf.int32)

    return features


@pytest.fixture
def tf_tabular_features(tabular_schema):
    return tf4rec.TabularFeatures.from_schema(
        tabular_schema,
        max_sequence_length=20,
        continuous_projection=64,
        aggregation="concat",
    )


@pytest.fixture
def tf_yoochoose_tabular_sequence_features(yoochoose_schema):
    return tf4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        # TODO: Add masking when we add it
        # masking="causal",
    )


def schema_like_generator(schema_file, lists_as_sequence_features):
    NUM_ROWS = 100
    DEFAULT_MAX_INT = 100
    MAX_SESSION_LENGTH = 20

    schema = Schema().from_proto_text(str(schema_file))
    schema = schema.remove_by_name(["session_id", "session_start", "day_idx"])
    data = {}

    for i in range(NUM_ROWS):
        session_length = random.randint(5, MAX_SESSION_LENGTH)

        for feature in schema.feature:
            is_list_feature = has_field(feature, "value_count")
            is_int_feature = has_field(feature, "int_domain")

            if is_int_feature:
                max_num = DEFAULT_MAX_INT
                if feature.int_domain.is_categorical:
                    max_num = feature.int_domain.max
                if is_list_feature:
                    list_length = (
                        session_length if lists_as_sequence_features else feature.value_count.max
                    )
                    row = tf.random.uniform((list_length,), maxval=max_num)
                else:
                    row = tf.random.uniform((1,), maxval=max_num)
            else:
                if is_list_feature:
                    list_length = (
                        session_length if lists_as_sequence_features else feature.value_count.max
                    )
                    row = tf.random.uniform((session_length,))
                else:
                    row = tf.random.uniform((1,))

            if is_list_feature:
                row = (row, [len(row)])

            if feature.name in data:
                if is_list_feature:
                    data[feature.name] = (
                        tf.concat((data[feature.name][0], row[0]), axis=0),
                        data[feature.name][1] + row[1],
                    )
                else:
                    data[feature.name] = tf.concat((data[feature.name], row), axis=0)
            else:
                data[feature.name] = row

    outputs = {}
    for key, val in data.items():
        if isinstance(val, tuple):
            offsets = [0]
            for length in val[1][:-1]:
                offsets.append(offsets[-1] + length)
            vals = (val[0], tf.expand_dims(tf.concat(offsets, axis=0), 1))
            values, offsets, diff_offsets, num_rows = _pull_values_offsets(vals)
            indices = _get_indices(offsets, diff_offsets)
            outputs[key] = _get_sparse_tensor(values, indices, num_rows, MAX_SESSION_LENGTH)
        else:
            outputs[key] = data[key]

    return outputs


@pytest.fixture
def tf_tabular_data(tabular_schema_file):
    return schema_like_generator(tabular_schema_file, lists_as_sequence_features=False)


@pytest.fixture
def tf_yoochoose_like(yoochoose_schema_file):
    return schema_like_generator(yoochoose_schema_file, lists_as_sequence_features=True)


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


@pytest.fixture
def tf_masking_inputs():
    # fixed parameters for tests
    NUM_EXAMPLES = 20
    MAX_LEN = 10
    PAD_TOKEN = 0
    hidden_dim = 16
    features = {}
    # generate random tensors for test
    features["input_tensor"] = tf.convert_to_tensor(
        np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim))
    )
    # create sequences
    labels = np.random.randint(1, MAX_CARDINALITY, (NUM_EXAMPLES, MAX_LEN))
    # replace last 2 items by zeros to mimic padding
    labels[:, MAX_LEN - 2 :] = 0
    labels = tf.convert_to_tensor(labels)
    features["labels"] = labels
    features["padding_idx"] = PAD_TOKEN
    features["vocab_size"] = MAX_CARDINALITY

    return features
