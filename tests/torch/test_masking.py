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

import numpy as np
import pytest

pytorch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers4rec.torch")

# fixed parameters for tests
lm_tasks = list(tr.masking.masking_registry.keys())


# Test output shapes
@pytest.mark.parametrize("task", lm_tasks)
def test_task_output_shape(torch_masking_inputs, task):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry[task](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    out = lm(torch_masking_inputs["input_tensor"], torch_masking_inputs["labels"], training=True)
    assert lm.masked_targets.shape[0] == torch_masking_inputs["input_tensor"].size(0)
    assert lm.masked_targets.shape[1] == torch_masking_inputs["input_tensor"].size(1)
    assert out.size(2) == torch_masking_inputs["input_tensor"].size(2)


# Test only last item is masked when evaluating
@pytest.mark.parametrize("task", lm_tasks)
def test_mask_only_last_item_for_eval(torch_masking_inputs, task):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry[task](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    lm.compute_masked_targets(torch_masking_inputs["labels"], training=False)
    # get non padded last items
    non_padded_mask = torch_masking_inputs["labels"] != torch_masking_inputs["padding_idx"]
    rows_ids = pytorch.arange(
        torch_masking_inputs["labels"].size(0),
        dtype=pytorch.long,
        device=torch_masking_inputs["labels"].device,
    )
    last_item_sessions = non_padded_mask.sum(axis=1) - 1
    last_labels = torch_masking_inputs["labels"][rows_ids, last_item_sessions].flatten().numpy()
    # get the last labels from output
    trgt_pad = lm.masked_targets != torch_masking_inputs["padding_idx"]
    out_last = lm.masked_targets[trgt_pad].flatten().numpy()
    # check that only one item is masked for each session
    assert lm.mask_schema.sum() == torch_masking_inputs["input_tensor"].size(0)
    # check only the last non-paded item is masked
    assert all(last_labels == out_last)


@pytest.mark.parametrize("task", lm_tasks)
def test_mask_all_next_item_for_eval(torch_masking_inputs, task):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry[task](
        hidden_dim,
        padding_idx=torch_masking_inputs["padding_idx"],
        eval_on_last_item_seq_only=False,
    )
    masking_info = lm.compute_masked_targets(torch_masking_inputs["labels"], training=False)
    # get the labels from output
    trgt_pad = masking_info.targets != torch_masking_inputs["padding_idx"]
    labels = masking_info.targets[trgt_pad].flatten().numpy()
    # get non padded items when shifting input sequence
    shift_inputs = torch_masking_inputs["labels"][:, 1:]
    non_padded_mask = shift_inputs != torch_masking_inputs["padding_idx"]
    n_labels_sessions = non_padded_mask.sum(axis=1)
    all_labels = shift_inputs[non_padded_mask].flatten().numpy()

    # check that number of labels per session matches
    assert all(masking_info.schema.sum(1) == n_labels_sessions)
    # check all next items are masked
    assert all(all_labels == labels)


# Test only last item is masked when training clm on last item
def test_clm_training_on_last_item(torch_masking_inputs):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry["causal"](
        hidden_dim,
        padding_idx=torch_masking_inputs["padding_idx"],
        train_on_last_item_seq_only=True,
    )
    lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    # get non padded last items
    non_padded_mask = torch_masking_inputs["labels"] != torch_masking_inputs["padding_idx"]
    rows_ids = pytorch.arange(
        torch_masking_inputs["labels"].size(0),
        dtype=pytorch.long,
        device=torch_masking_inputs["labels"].device,
    )
    last_item_sessions = non_padded_mask.sum(axis=1) - 1
    last_labels = torch_masking_inputs["labels"][rows_ids, last_item_sessions].flatten().numpy()
    # last labels from output
    trgt_pad = lm.masked_targets != torch_masking_inputs["padding_idx"]
    out_last = lm.masked_targets[trgt_pad].flatten().numpy()
    assert lm.mask_schema.sum() == torch_masking_inputs["input_tensor"].size(0)
    assert all(last_labels == out_last)


# Test at least one item is masked when training
@pytest.mark.parametrize("task", lm_tasks)
def test_at_least_one_masked_item_mlm(torch_masking_inputs, task):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry[task](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    masking_info = lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    trgt_mask = masking_info.targets != torch_masking_inputs["padding_idx"]
    assert all(trgt_mask.sum(axis=1).numpy() > 0)


# Check that not all items are masked when training
@pytest.mark.parametrize("task", lm_tasks)
def test_not_all_masked_lm(torch_masking_inputs, task):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry[task](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    trgt_mask = lm.masked_targets != torch_masking_inputs["padding_idx"]
    non_padded_mask = torch_masking_inputs["labels"] != torch_masking_inputs["padding_idx"]
    assert all(trgt_mask.sum(axis=1).numpy() != non_padded_mask.sum(axis=1).numpy())


# Check number of masked positions equal to number of targets
@pytest.mark.parametrize("task", lm_tasks)
def test_task_masked_cardinality(torch_masking_inputs, task):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry[task](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    trgt_pad = lm.masked_targets != torch_masking_inputs["padding_idx"]
    assert lm.mask_schema.sum() == trgt_pad.sum()


# Check target mapping are not None when PLM
def test_plm_output_shape(torch_masking_inputs):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry["permutation"](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    assert lm.target_mapping is not None
    assert lm.perm_mask is not None


# Check that only masked items are replaced
def test_replaced_fake_tokens(torch_masking_inputs):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.masking_registry["replacement"](
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"]
    )
    masking_info = lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    trg_flat = masking_info.targets.flatten()
    non_pad_mask = trg_flat != torch_masking_inputs["padding_idx"]
    # Nb of pos items
    pos_items = non_pad_mask.sum()
    # generate random logits
    logits = pytorch.tensor(
        np.random.uniform(0, 1, (pos_items, torch_masking_inputs["vocab_size"]))
    )
    corrupted_inputs, discriminator_labels, _ = lm.get_fake_tokens(
        torch_masking_inputs["labels"], trg_flat, logits
    )
    replacement_mask = discriminator_labels != 0
    assert all(lm.mask_schema[replacement_mask] == 1)


# check fake replacements are items from the same batch when sample_from_batch is True
def test_replacement_from_batch(torch_masking_inputs):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.ReplacementLanguageModeling(
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"], sample_from_batch=True
    )
    masking_info = lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)

    trg_flat = masking_info.targets.flatten()
    non_pad_mask = trg_flat != torch_masking_inputs["padding_idx"]
    # Nb of pos items
    pos_items = non_pad_mask.sum()
    # generate random logits
    logits = pytorch.tensor(np.random.uniform(0, 1, (pos_items, pos_items)))
    corrupted_inputs, discriminator_labels, updates = lm.get_fake_tokens(
        torch_masking_inputs["labels"], trg_flat, logits
    )
    replacement_mask = discriminator_labels != 0
    assert set(corrupted_inputs[replacement_mask].numpy()).issubset(trg_flat[non_pad_mask].numpy())


# check output shape of sample_from_softmax
def test_sample_from_softmax_output(torch_masking_inputs):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.ReplacementLanguageModeling(
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"], sample_from_batch=True
    )
    masking_info = lm.compute_masked_targets(torch_masking_inputs["labels"], training=True)
    trg_flat = masking_info.targets.flatten()
    non_pad_mask = trg_flat != torch_masking_inputs["padding_idx"]
    # Nb of pos items
    pos_items = non_pad_mask.sum()
    # generate random logits
    logits = pytorch.tensor(np.random.uniform(0, 1, (pos_items, pos_items)))
    updates = lm.sample_from_softmax(logits)
    assert updates.size(0) == pos_items


@pytest.mark.parametrize("mode", [True, False])
def test_masked_positions(torch_masking_inputs, mode):
    hidden_dim = torch_masking_inputs["input_tensor"].size(2)
    lm = tr.masking.ReplacementLanguageModeling(
        hidden_dim, padding_idx=torch_masking_inputs["padding_idx"], sample_from_batch=True
    )
    masking_info = lm.compute_masked_targets(torch_masking_inputs["labels"], training=mode)

    targets = pytorch.masked_select(masking_info.targets, masking_info.schema)
    assert all(targets != torch_masking_inputs["padding_idx"])
