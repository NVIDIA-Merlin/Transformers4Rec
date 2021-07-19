import pytest
import numpy as np
from transformers4rec.torch.masking import masking_tasks
from transformers4rec.torch.masking import PermutationLanguageModeling

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")

MAX_LEN = 5
NUM_EXAMPLES = 8
PAD_TOKEN = 0
VOCAB_SIZE = 100


# Test output shapes
@pytest.mark.parametrize("task", ["masked", "causal", "permutation", "replacement"])
def test_task_output_shape(task):
    hidden_dim = 16
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    assert out.masked_label.shape[0] == NUM_EXAMPLES
    assert out.masked_label.shape[1] == MAX_LEN
    assert out.masked_input.size(2) == hidden_dim


# Test only last item is masked when evaluating
@pytest.mark.parametrize("task", ["masked", "causal", "permutation", "replacement"])
def test_mlm_eval(task):
    hidden_dim = 16
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
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


# Test at least one item is masked when training
@pytest.mark.parametrize("task", ["masked", "causal", "permutation", "replacement"])
def test_at_least_one_masked_item_mlm(task):
    hidden_dim = 16
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    trgt_mask = out.masked_label != PAD_TOKEN
    assert all(trgt_mask.sum(axis=1).numpy() > 0)


# Check that not all items are masked when training
@pytest.mark.parametrize("task", ["masked", "causal", "permutation", "replacement"])
def test_not_all_masked_mlm(task):
    hidden_dim = 16
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
    lm = masking_tasks[task](hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    trgt_mask = out.masked_label != PAD_TOKEN
    non_padded_mask = labels != PAD_TOKEN
    assert all(trgt_mask.sum(axis=1).numpy() != non_padded_mask.sum(axis=1).numpy())


# Check target mapping are not None when PLM
def test_plm_output_shape():
    hidden_dim = 16
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
    lm = PermutationLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = lm(input_tensor, labels, training=True)
    assert out.plm_target_mapping is not None
    assert out.plm_perm_mask is not None
