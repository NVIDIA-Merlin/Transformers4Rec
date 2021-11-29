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

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
if tf:

    def enable_tf_gpus_memory_growth():
        """Sets the GPU memory to be allocated as need by TF, so that TF does not
        reserve all GPU memory and causes OOM errors on the PyTorch tests
        """
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    enable_tf_gpus_memory_growth()

tr = pytest.importorskip("transformers4rec.tf")
schema_utils = pytest.importorskip("transformers4rec.tf.utils.schema_utils")

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
    return tr.TabularFeatures.from_schema(
        tabular_schema,
        max_sequence_length=20,
        continuous_projection=64,
        aggregation="concat",
    )


@pytest.fixture
def tf_yoochoose_tabular_sequence_features(yoochoose_schema):
    return tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )


@pytest.fixture
def tf_tabular_data():
    return tr.data.tabular_testing_data.tf_synthetic_data(num_rows=100)


@pytest.fixture
def tf_yoochoose_like():
    return tr.data.tabular_sequence_testing_data.tf_synthetic_data(
        num_rows=100, min_session_length=5, max_session_length=20
    )


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


@pytest.fixture
def tf_ranking_metrics_inputs():
    POS_EXAMPLE = 30
    VOCAB_SIZE = 40
    features = {}
    features["scores"] = tf.convert_to_tensor(np.random.uniform(0, 1, (POS_EXAMPLE, VOCAB_SIZE)))
    features["ks"] = tf.convert_to_tensor([1, 2, 3, 5, 10, 20])
    features["labels_one_hot"] = tf.convert_to_tensor(
        np.random.choice(a=[0, 1], size=(POS_EXAMPLE, VOCAB_SIZE))
    )

    features["labels"] = tf.convert_to_tensor(np.random.randint(1, VOCAB_SIZE, (POS_EXAMPLE,)))
    return features


@pytest.fixture()
def tf_next_item_prediction_model(yoochoose_schema):
    def get_model(masking="causal", schema=yoochoose_schema, d_output=64, config=tr.XLNetConfig):
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=d_output,
            masking=masking,
        )

        transformer_config = config.build(
            d_model=d_output, n_head=8, n_layer=2, total_seq_length=20
        )
        body = tr.SequentialBlock(
            [
                input_module,
                tr.TransformerBlock(transformer_config, masking=input_module.masking),
            ]
        )
        task = tr.NextItemPredictionTask(weight_tying=True)
        model = task.to_model(body=body)
        return model

    return get_model
