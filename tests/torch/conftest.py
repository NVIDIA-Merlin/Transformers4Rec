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

import pytest

pytorch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
tr = pytest.importorskip("transformers4rec.torch")
schema_utils = pytest.importorskip("transformers4rec.torch.utils.schema_utils")

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
def torch_tabular_features():
    return tr.TabularFeatures.from_schema(
        tr.data.tabular_testing_data.schema,
        max_sequence_length=20,
        continuous_projection=64,
        aggregation="concat",
    )


@pytest.fixture
def torch_yoochoose_tabular_transformer_features():
    return tr.TabularSequenceFeatures.from_schema(
        tr.data.tabular_sequence_testing_data.schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )


@pytest.fixture
def torch_yoochoose_next_item_prediction_model(torch_yoochoose_tabular_transformer_features):
    # define Transformer-based model
    inputs = torch_yoochoose_tabular_transformer_features
    transformer_config = tr.XLNetConfig.build(d_model=64, n_head=4, n_layer=2, total_seq_length=20)
    body = tr.SequentialBlock(
        inputs,
        tr.MLPBlock([64]),
        tr.TransformerBlock(transformer_config, masking=inputs.masking),
    )
    model = tr.NextItemPredictionTask(weight_tying=True, hf_format=True).to_model(body, inputs)
    return model


@pytest.fixture
def torch_tabular_data():
    return tr.data.tabular_testing_data.torch_synthetic_data(num_rows=100)


@pytest.fixture
def torch_yoochoose_like():
    return tr.data.tabular_sequence_testing_data.torch_synthetic_data(
        num_rows=100, min_session_length=5, max_session_length=20
    )
