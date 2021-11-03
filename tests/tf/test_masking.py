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

tf = pytest.importorskip("tensorflow")
tr = pytest.importorskip("transformers4rec.tf")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")

lm_tasks = list(tr.masking.masking_registry.keys())


# Test output shapes
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_task_output_shape(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](padding_idx=tf_masking_inputs["padding_idx"])
    out = lm(tf_masking_inputs["input_tensor"], tf_masking_inputs["labels"], training=True)
    assert tf.shape(lm.masked_targets)[0] == tf_masking_inputs["input_tensor"].shape[0]
    assert tf.shape(lm.masked_targets)[1] == tf_masking_inputs["input_tensor"].shape[1]
    assert out.shape[2] == tf_masking_inputs["input_tensor"].shape[2]


# Test class serialization
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_serialization_masking(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](padding_idx=tf_masking_inputs["padding_idx"])
    copy_layer = test_utils.assert_serialization(lm)

    out = copy_layer(tf_masking_inputs["input_tensor"], tf_masking_inputs["labels"], training=True)
    assert tf.shape(copy_layer.masked_targets)[0] == tf_masking_inputs["input_tensor"].shape[0]
    assert out.shape[2] == tf_masking_inputs["input_tensor"].shape[2]


# Test eager + graph modes
@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_masking_model(yoochoose_schema, tf_yoochoose_like, run_eagerly, task):
    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking=task,
    )
    body = tr.SequentialBlock([input_module, tr.MLPBlock([64])])
    test_utils.assert_body_works_in_model(tf_yoochoose_like, input_module, body, run_eagerly)


# Test only last item is masked when eval_on_last_item_seq_only
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_mask_only_last_item_for_eval(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](
        padding_idx=tf_masking_inputs["padding_idx"], eval_on_last_item_seq_only=True
    )
    lm.compute_masked_targets(tf_masking_inputs["labels"], training=False)
    # get non padded last items
    non_padded_mask = tf_masking_inputs["labels"] != tf_masking_inputs["padding_idx"]
    rows_ids = tf.range(tf_masking_inputs["labels"].shape[0], dtype=tf.int64)
    last_item_sessions = tf.reduce_sum(tf.cast(non_padded_mask, tf.int64), axis=1) - 1
    indices = tf.concat(
        [tf.expand_dims(rows_ids, 1), tf.expand_dims(last_item_sessions, 1)], axis=1
    )
    last_labels = tf.gather_nd(tf_masking_inputs["labels"], indices).numpy()
    # get the last labels from output
    trgt_pad = lm.masked_targets != tf_masking_inputs["padding_idx"]
    out_last = tf.boolean_mask(lm.masked_targets, trgt_pad).numpy()

    # check that only one item is masked for each session
    assert (
        tf.reduce_sum(tf.cast(lm.mask_schema, tf.int32)).numpy()
        == tf_masking_inputs["input_tensor"].shape[0]
    )

    # check only the last non-paded item is masked
    assert all(last_labels == out_last)


@pytest.mark.parametrize("task", ["causal", "masked"])
def test_mask_all_next_item_for_eval(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](
        padding_idx=tf_masking_inputs["padding_idx"],
        eval_on_last_item_seq_only=False,
    )
    masking_info = lm.compute_masked_targets(tf_masking_inputs["labels"], training=False)
    # get the labels from output
    trgt_pad = masking_info.targets != tf_masking_inputs["padding_idx"]
    labels = masking_info.targets[trgt_pad].numpy()
    # get non padded items when shifting input sequence
    shift_inputs = tf_masking_inputs["labels"][:, 1:]
    non_padded_mask = shift_inputs != tf_masking_inputs["padding_idx"]
    n_labels_sessions = non_padded_mask.numpy().sum(1)
    all_labels = tf.boolean_mask(shift_inputs, non_padded_mask).numpy()

    # check that number of labels per session matches
    assert all(masking_info.schema.numpy().sum(1) == n_labels_sessions)
    # check all next items are masked
    assert all(all_labels == labels)


# Test at least one item is masked when training
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_at_least_one_masked_item_mlm(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](padding_idx=tf_masking_inputs["padding_idx"])
    masking_info = lm.compute_masked_targets(tf_masking_inputs["labels"], training=True)
    trgt_mask = tf.cast(masking_info.targets != tf_masking_inputs["padding_idx"], tf.int32)
    assert all(tf.reduce_sum(trgt_mask, axis=1).numpy() > 0)


# Check that not all items are masked when training
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_not_all_masked_lm(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](padding_idx=tf_masking_inputs["padding_idx"])
    lm.compute_masked_targets(tf_masking_inputs["labels"], training=True)
    trgt_mask = lm.masked_targets != tf_masking_inputs["padding_idx"]
    non_padded_mask = tf_masking_inputs["labels"] != tf_masking_inputs["padding_idx"]
    assert all(trgt_mask.numpy().sum(axis=1) != non_padded_mask.numpy().sum(axis=1))


# Check number of masked positions equal to number of targets
@pytest.mark.parametrize("task", ["causal", "masked"])
def test_task_masked_cardinality(tf_masking_inputs, task):
    lm = tr.masking.masking_registry[task](padding_idx=tf_masking_inputs["padding_idx"])
    lm.compute_masked_targets(tf_masking_inputs["labels"], training=True)
    trgt_pad = lm.masked_targets != tf_masking_inputs["padding_idx"]
    assert lm.mask_schema.numpy().sum() == trgt_pad.numpy().sum()


# Test only last item is masked when training clm on last item
def test_clm_training_on_last_item(tf_masking_inputs):
    lm = tr.masking.masking_registry["causal"](
        padding_idx=tf_masking_inputs["padding_idx"],
        train_on_last_item_seq_only=True,
    )
    lm.compute_masked_targets(tf_masking_inputs["labels"], training=True)
    # get non padded last items
    non_padded_mask = tf_masking_inputs["labels"] != tf_masking_inputs["padding_idx"]
    rows_ids = tf.range(tf_masking_inputs["labels"].shape[0], dtype=tf.int64)
    last_item_sessions = tf.reduce_sum(tf.cast(non_padded_mask, tf.int64), axis=1) - 1
    indices = tf.concat(
        [tf.expand_dims(rows_ids, 1), tf.expand_dims(last_item_sessions, 1)], axis=1
    )
    last_labels = tf.gather_nd(tf_masking_inputs["labels"], indices).numpy()
    # get the last labels from output
    trgt_pad = lm.masked_targets != tf_masking_inputs["padding_idx"]
    out_last = tf.boolean_mask(lm.masked_targets, trgt_pad).numpy()

    # check that only one item is masked for each session
    assert (
        tf.reduce_sum(tf.cast(lm.mask_schema, tf.int32)).numpy()
        == tf_masking_inputs["input_tensor"].shape[0]
    )

    # check only the last non-paded item is masked
    assert all(last_labels == out_last)
