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

from merlin_standard_lib import Tag, schema

tf = pytest.importorskip("tensorflow")
tr = pytest.importorskip("transformers4rec.tf")


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise(replacement_prob):
    NUM_SEQS = 100
    SEQ_LENGTH = 80
    PAD_TOKEN = 0

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    seq_inputs = {
        "categ_seq_feat": tf.experimental.numpy.tril(
            tf.random.uniform((NUM_SEQS, SEQ_LENGTH), minval=1, maxval=100, dtype=tf.int32), 1
        ),
        "cont_seq_feat": tf.experimental.numpy.tril(tf.random.uniform((NUM_SEQS, SEQ_LENGTH)), 1),
        "categ_feat": tf.random.uniform((NUM_SEQS,), minval=1, maxval=100, dtype=tf.int32),
    }

    tf.random.set_seed(0)
    ssn = tr.StochasticSwapNoise(pad_token=PAD_TOKEN, replacement_prob=replacement_prob)
    mask = seq_inputs["categ_seq_feat"] != PAD_TOKEN
    out_features_ssn = ssn(seq_inputs, input_mask=mask)

    for fname in seq_inputs:
        replaced_mask = out_features_ssn[fname] != seq_inputs[fname]
        replaced_mask_non_padded = tf.boolean_mask(replaced_mask, seq_inputs[fname] != PAD_TOKEN)
        replacement_rate = tf.reduce_mean(
            tf.cast(replaced_mask_non_padded, dtype=tf.float32)
        ).numpy()
        assert replacement_rate == pytest.approx(replacement_prob, abs=0.15)


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise_disable_on_eval(replacement_prob):
    NUM_SEQS = 100
    SEQ_LENGTH = 80
    PAD_TOKEN = 0

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    seq_inputs = {
        "categ_seq_feat": tf.experimental.numpy.tril(
            tf.random.uniform((NUM_SEQS, SEQ_LENGTH), minval=1, maxval=100, dtype=tf.int32), 1
        ),
        "cont_seq_feat": tf.experimental.numpy.tril(tf.random.uniform((NUM_SEQS, SEQ_LENGTH)), 1),
        "categ_feat": tf.random.uniform((NUM_SEQS,), minval=1, maxval=100, dtype=tf.int32),
    }

    tf.random.set_seed(0)
    ssn = tr.StochasticSwapNoise(pad_token=PAD_TOKEN, replacement_prob=replacement_prob)
    mask = seq_inputs["categ_seq_feat"] != PAD_TOKEN
    out_features_ssn = ssn(seq_inputs, input_mask=mask, training=False)

    for fname in seq_inputs:
        replaced_mask = out_features_ssn[fname] == seq_inputs[fname]
        assert tf.reduce_all(replaced_mask).numpy()


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise_with_tabular_features(
    yoochoose_schema, tf_yoochoose_like, replacement_prob
):
    inputs = tf_yoochoose_like
    tab_module = tr.TabularSequenceFeatures.from_schema(yoochoose_schema)
    out_features = tab_module(inputs)

    PAD_TOKEN = 0
    tf.random.set_seed(0)
    ssn = tr.StochasticSwapNoise(
        pad_token=PAD_TOKEN, replacement_prob=replacement_prob, schema=yoochoose_schema
    )

    out_features_ssn = tab_module(inputs, pre=ssn)

    for fname in out_features_ssn:
        replaced_mask = out_features[fname] != out_features_ssn[fname]

        # Ignoring padding items to compute the mean replacement rate
        feat_non_padding_mask = inputs[fname] != PAD_TOKEN
        replaced_mask_non_padded = tf.boolean_mask(replaced_mask, feat_non_padding_mask)
        replacement_rate = tf.reduce_mean(
            tf.cast(replaced_mask_non_padded, dtype=tf.float32)
        ).numpy()
        assert replacement_rate == pytest.approx(replacement_prob, abs=0.15)


def test_stochastic_swap_noise_raise_exception_not_2d_item_id():

    s = schema.Schema(
        [
            schema.ColumnSchema.create_categorical(
                "item_id_feat", num_items=1000, tags=[Tag.ITEM_ID]
            ),
        ]
    )

    NUM_SEQS = 100
    SEQ_LENGTH = 80
    PAD_TOKEN = 0

    seq_inputs = {
        "item_id_feat": tf.experimental.numpy.tril(
            tf.random.uniform((NUM_SEQS, SEQ_LENGTH, 64), minval=1, maxval=100, dtype=tf.int32), 1
        ),
    }

    ssn = tr.StochasticSwapNoise(pad_token=PAD_TOKEN, replacement_prob=0.3, schema=s)

    with pytest.raises(ValueError) as excinfo:
        ssn(seq_inputs)
    assert "To extract the padding mask from item id tensor it is expected to have 2 dims" in str(
        excinfo.value
    )
