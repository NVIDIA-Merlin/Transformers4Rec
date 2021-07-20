import numpy as np
import pytest

from transformers4rec.torch.masking import (
    PermutationLanguageModeling,
    ReplacementLanguageModeling,
    masking_tasks,
)

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")

# fixed parameters for tests
MAX_LEN = 10
NUM_EXAMPLES = 8
PAD_TOKEN = 0
VOCAB_SIZE = 100
hidden_dim = 16
masking_taks = ["masked", "causal", "permutation", "replacement"]

# generate random tensors for test
input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
# create sequences
labels = torch.tensor(np.random.randint(1, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
# replace last 2 items by zeros to mimic padding
labels[:, MAX_LEN - 2 :] = 0


# Test output shapes
@pytest.mark.parametrize("task", masking_taks)
def test_task_output_shape(task):
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    assert out.masked_label.shape[0] == NUM_EXAMPLES
    assert out.masked_label.shape[1] == MAX_LEN
    assert out.masked_input.size(2) == hidden_dim


# Test only last item is masked when evaluating
@pytest.mark.parametrize("task", masking_taks)
def test_mlm_eval(task):
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=False)
    assert out.mask_schema.sum() == NUM_EXAMPLES
    # get non padded last items
    non_padded_mask = labels != PAD_TOKEN
    rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
    last_item_sessions = non_padded_mask.sum(axis=1) - 1
    last_labels = labels[rows_ids, last_item_sessions].flatten().numpy()
    # last labels from output
    trgt_pad = out.masked_label != PAD_TOKEN
    out_last = out.masked_label[trgt_pad].flatten().numpy()
    assert all(last_labels == out_last)


# Test only last item is masked when training clm on last item
def test_clm_training_on_last_item():
    lm = masking_tasks["causal"](hidden_dim, pad_token=PAD_TOKEN, train_on_last_item_seq_only=True)
    out = lm(input_tensor, labels, training=True)
    assert out.mask_schema.sum() == NUM_EXAMPLES
    # get non padded last items
    non_padded_mask = labels != PAD_TOKEN
    rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
    last_item_sessions = non_padded_mask.sum(axis=1) - 1
    last_labels = labels[rows_ids, last_item_sessions].flatten().numpy()
    # last labels from output
    trgt_pad = out.masked_label != PAD_TOKEN
    out_last = out.masked_label[trgt_pad].flatten().numpy()
    assert all(last_labels == out_last)


# Test at least one item is masked when training
@pytest.mark.parametrize("task", masking_taks)
def test_at_least_one_masked_item_mlm(task):
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    trgt_mask = out.masked_label != PAD_TOKEN
    assert all(trgt_mask.sum(axis=1).numpy() > 0)


# Check that not all items are masked when training
@pytest.mark.parametrize("task", masking_taks)
def test_not_all_masked_mlm(task):
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    trgt_mask = out.masked_label != PAD_TOKEN
    non_padded_mask = labels != PAD_TOKEN
    assert all(trgt_mask.sum(axis=1).numpy() != non_padded_mask.sum(axis=1).numpy())


# Check number of masked positions equal to number of targets
@pytest.mark.parametrize("task", ["masked", "causal", "replacement"])
def test_task_masked_cardinality(task):
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    trgt_pad = out.masked_label != PAD_TOKEN
    assert out.mask_schema.sum() == trgt_pad.sum()


# Check target mapping are not None when PLM
def test_plm_output_shape():
    lm = PermutationLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    assert out.plm_target_mapping is not None
    assert out.plm_perm_mask is not None


# Check that only masked items are replaced
def test_replaced_fake_tokens():
    lm = ReplacementLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    trg_flat = out.masked_label.flatten()
    non_pad_mask = trg_flat != PAD_TOKEN
    # Nb of pos items
    pos_items = non_pad_mask.sum()
    # generate random logits
    logits = torch.tensor(np.random.uniform(0, 1, (pos_items, VOCAB_SIZE)))
    corrupted_inputs, discriminator_labels, _ = lm.get_fake_tokens(labels, trg_flat, logits)
    replacement_mask = discriminator_labels != 0
    assert all(out.mask_schema[replacement_mask] == 1)


# check fake replacement are from items of same batch
def test_replacement_from_batch():
    lm = ReplacementLanguageModeling(hidden_dim, pad_token=PAD_TOKEN, sample_from_batch=True)
    out = lm(input_tensor, labels, training=True)
    trg_flat = out.masked_label.flatten()
    non_pad_mask = trg_flat != PAD_TOKEN
    # Nb of pos items
    pos_items = non_pad_mask.sum()
    # generate random logits
    logits = torch.tensor(np.random.uniform(0, 1, (pos_items, pos_items)))
    corrupted_inputs, discriminator_labels, updates = lm.get_fake_tokens(labels, trg_flat, logits)
    replacement_mask = discriminator_labels != 0
    assert set(corrupted_inputs[replacement_mask].numpy()).issubset(trg_flat[non_pad_mask].numpy())


# check output shape of sample_from_softmax
def test_sample_from_softmax_output():
    lm = ReplacementLanguageModeling(hidden_dim, pad_token=PAD_TOKEN, sample_from_batch=True)
    out = lm(input_tensor, labels, training=True)
    trg_flat = out.masked_label.flatten()
    non_pad_mask = trg_flat != PAD_TOKEN
    # Nb of pos items
    pos_items = non_pad_mask.sum()
    # generate random logits
    logits = torch.tensor(np.random.uniform(0, 1, (pos_items, pos_items)))
    updates = lm.sample_from_softmax(logits)
    assert updates.size(0) == pos_items
