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

import pytest

from merlin_standard_lib import Schema
from merlin_standard_lib.utils.proto_utils import has_field
from transformers4rec.config import transformer as tconf

pytorch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
torch4rec = pytest.importorskip("transformers4rec.torch")

ASSETS_DIR = pathlib.Path(__file__).parent.parent / "assets"

NUM_EXAMPLES = 1000
MAX_CARDINALITY = 100


@pytest.fixture
def torch_con_features():
    features = {}
    keys = [f"con_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = pytorch.rand((NUM_EXAMPLES, 1))

    return features


@pytest.fixture
def torch_cat_features():
    features = {}
    keys = [f"cat_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = pytorch.randint(MAX_CARDINALITY, (NUM_EXAMPLES,))

    return features


@pytest.fixture
def torch_masking_inputs():
    # fixed parameters for tests
    NUM_EXAMPLES = 20
    MAX_LEN = 10
    PAD_TOKEN = 0
    hidden_dim = 16
    features = {}
    # generate random tensors for test
    features["input_tensor"] = pytorch.tensor(
        np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim))
    )
    # create sequences
    labels = pytorch.tensor(np.random.randint(1, MAX_CARDINALITY, (NUM_EXAMPLES, MAX_LEN)))
    # replace last 2 items by zeros to mimic padding
    labels[:, MAX_LEN - 2 :] = 0
    features["labels"] = labels
    features["padding_idx"] = PAD_TOKEN
    features["vocab_size"] = MAX_CARDINALITY

    return features


@pytest.fixture
def torch_seq_prediction_head_inputs():
    ITEM_DIM = 128
    POS_EXAMPLE = 25
    features = {}
    features["seq_model_output"] = pytorch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, ITEM_DIM)))
    features["item_embedding_table"] = pytorch.nn.Embedding(MAX_CARDINALITY, ITEM_DIM)
    features["labels_all"] = pytorch.tensor(np.random.randint(1, MAX_CARDINALITY, (POS_EXAMPLE,)))
    features["vocab_size"] = MAX_CARDINALITY
    features["item_dim"] = ITEM_DIM
    return features


@pytest.fixture
def torch_ranking_metrics_inputs():
    POS_EXAMPLE = 30
    VOCAB_SIZE = 40
    features = {}
    features["scores"] = pytorch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, VOCAB_SIZE)))
    features["ks"] = pytorch.LongTensor([1, 2, 3, 5, 10, 20])
    features["labels_one_hot"] = pytorch.LongTensor(
        np.random.choice(a=[0, 1], size=(POS_EXAMPLE, VOCAB_SIZE))
    )

    features["labels"] = pytorch.tensor(np.random.randint(1, VOCAB_SIZE, (POS_EXAMPLE,)))
    return features


@pytest.fixture
def torch_seq_prediction_head_link_to_block():
    ITEM_DIM = 64
    POS_EXAMPLE = 25
    features = {}
    features["seq_model_output"] = pytorch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, ITEM_DIM)))
    features["item_embedding_table"] = pytorch.nn.Embedding(MAX_CARDINALITY, ITEM_DIM)
    features["labels_all"] = pytorch.tensor(np.random.randint(1, MAX_CARDINALITY, (POS_EXAMPLE,)))
    features["vocab_size"] = MAX_CARDINALITY
    features["item_dim"] = ITEM_DIM
    features["config"] = {
        "item": {
            "dtype": "categorical",
            "cardinality": MAX_CARDINALITY,
            "tags": ["categorical", "item"],
            "log_as_metadata": True,
        }
    }

    return features


@pytest.fixture
def torch_tabular_features(tabular_schema):
    return torch4rec.TabularFeatures.from_schema(
        tabular_schema,
        max_sequence_length=20,
        continuous_projection=64,
        aggregation="concat",
    )


@pytest.fixture
def torch_yoochoose_tabular_transformer_features(yoochoose_schema):
    return torch4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )


@pytest.fixture
def torch_yoochoose_next_item_prediction_model(torch_yoochoose_tabular_transformer_features):
    # define Transformer-based model
    inputs = torch_yoochoose_tabular_transformer_features
    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    body = torch4rec.SequentialBlock(
        inputs,
        torch4rec.MLPBlock([64]),
        torch4rec.TransformerBlock(transformer_config, masking=inputs.masking),
    )
    model = torch4rec.NextItemPredictionTask(weight_tying=True, hf_format=True).to_model(
        body, inputs
    )
    return model


def schema_like_generator(schema_file, lists_as_sequence_features):
    NUM_ROWS = 100
    DEFAULT_MAX_INT = 100
    MAX_LIST_LENGTH = 20

    schema = Schema().from_proto_text(str(schema_file))
    schema = schema.remove_by_name(["session_id", "session_start", "day_idx"])
    data = {}

    for i in range(NUM_ROWS):
        if lists_as_sequence_features:
            session_length = random.randint(2, MAX_LIST_LENGTH)

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
                    row = pytorch.randint(1, max_num, (list_length,))

                else:
                    row = pytorch.randint(1, max_num, (1,))
            else:
                if is_list_feature:
                    list_length = (
                        session_length if lists_as_sequence_features else feature.value_count.max
                    )
                    row = pytorch.rand((list_length,))
                else:
                    row = pytorch.rand((1,))

            if is_list_feature:
                row = (row, [len(row)])

            if feature.name in data:
                if is_list_feature:
                    data[feature.name] = (
                        pytorch.cat((data[feature.name][0], row[0])),
                        data[feature.name][1] + row[1],
                    )
                else:
                    data[feature.name] = pytorch.cat((data[feature.name], row))
            else:
                data[feature.name] = row

    outputs = {}
    for key, val in data.items():
        if isinstance(val, tuple):
            offsets = [0]
            for length in val[1][:-1]:
                offsets.append(offsets[-1] + length)
            vals = (val[0], pytorch.tensor(offsets).unsqueeze(dim=1))
            values, offsets, diff_offsets, num_rows = _pull_values_offsets(vals)
            indices = _get_indices(offsets, diff_offsets)
            outputs[key] = _get_sparse_tensor(values, indices, num_rows, MAX_LIST_LENGTH)
        else:
            outputs[key] = data[key]

    return outputs


@pytest.fixture
def torch_tabular_data(tabular_schema_file):
    outputs = schema_like_generator(tabular_schema_file, lists_as_sequence_features=False)
    return outputs


@pytest.fixture
def torch_yoochoose_like(yoochoose_schema_file):
    return schema_like_generator(yoochoose_schema_file, lists_as_sequence_features=True)


def _pull_values_offsets(values_offset):
    # pull_values_offsets, return values offsets diff_offsets
    if isinstance(values_offset, tuple):
        values = values_offset[0].flatten()
        offsets = values_offset[1].flatten()
    else:
        values = values_offset.flatten()
        offsets = pytorch.arange(values.size()[0])
    num_rows = len(offsets)
    offsets = pytorch.cat([offsets, pytorch.tensor([len(values)])])
    diff_offsets = offsets[1:] - offsets[:-1]
    return values, offsets, diff_offsets, num_rows


def _get_indices(offsets, diff_offsets):
    row_ids = pytorch.arange(len(offsets) - 1)
    row_ids_repeated = pytorch.repeat_interleave(row_ids, diff_offsets)
    row_offset_repeated = pytorch.repeat_interleave(offsets[:-1], diff_offsets)
    col_ids = pytorch.arange(len(row_offset_repeated)) - row_offset_repeated
    indices = pytorch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices


def _get_sparse_tensor(values, indices, num_rows, seq_limit):
    sparse_tensor = pytorch.sparse_coo_tensor(
        indices.T, values, pytorch.Size([num_rows, seq_limit])
    )

    return sparse_tensor.to_dense()
