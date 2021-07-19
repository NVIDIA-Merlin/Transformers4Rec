import pytest
import torch
import numpy as np 
from transformers4rec.torch.masking import MaskedLanguageModeling, PermutationLanguageModeling


torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")

MAX_LEN = 5
NUM_EXAMPLES = 8
PAD_TOKEN = 0

# Test output shapes 
def test_mlm_output_shape():
    hidden_dim = 16 
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, 100, (NUM_EXAMPLES, MAX_LEN)))
    mlm = MaskedLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = mlm(input_tensor, labels, training=True)
    assert out.masked_label.shape[0] == NUM_EXAMPLES
    assert out.masked_label.shape[1] == MAX_LEN
    assert out.masked_input.size(2) == hidden_dim 
    
# Test number of masked positions 
def test_mlm_masked_cardinality():
    hidden_dim = 16 
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, 100, (NUM_EXAMPLES, MAX_LEN)))
    mlm = mlm = MaskedLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = mlm(input_tensor, labels, training=True)
    trgt_pad = out.masked_label != PAD_TOKEN
    assert out.mask_schema.sum() == trgt_pad.sum()

    
#test only last item is masked when evaluating    
def test_mlm_eval():
    hidden_dim = 16 
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, 100, (NUM_EXAMPLES, MAX_LEN)))
    mlm = mlm = MaskedLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = mlm(input_tensor, labels, training=False)
    assert out.mask_schema.sum() == NUM_EXAMPLES
    #get non padded last items 
    non_padded_mask = labels != PAD_TOKEN
    rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
    last_item_sessions = non_padded_mask.sum(axis=1) - 1
    last_labels = labels[rows_ids, last_item_sessions].flatten().numpy()
    # last labels from output 
    trgt_pad = out.masked_label != PAD_TOKEN
    out_last = out.masked_label[trgt_pad].flatten().numpy()
    assert all(last_labels == out_last )

    
#Test at least one item is masked when training 
def test_at_least_one_masked_item_mlm(): 
    hidden_dim = 16 
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, 100, (NUM_EXAMPLES, MAX_LEN)))
    mlm = mlm = MaskedLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = mlm(input_tensor, labels, training=True)
    trgt_mask = out.masked_label != PAD_TOKEN
    assert all(trgt_mask.sum(axis=1).numpy() > 0 )
    
#Check that not all items are masked when training 
def test_not_all_masked_mlm():
    hidden_dim = 16 
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, 100, (NUM_EXAMPLES, MAX_LEN)))
    mlm = mlm = MaskedLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = mlm(input_tensor, labels, training=True)
    trgt_mask = out.masked_label != PAD_TOKEN
    non_padded_mask = labels != PAD_TOKEN
    assert all(trgt_mask.sum(axis=1).numpy() != non_padded_mask.sum(axis=1).numpy())
    
    
#check target mapping are not None when PLM  
def test_plm_output_shape():
    hidden_dim = 16 
    input_tensor = torch.tensor(np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim)))
    labels = torch.tensor(np.random.randint(0, 100, (NUM_EXAMPLES, MAX_LEN)))
    mlm = PermutationLanguageModeling(hidden_dim, pad_token=PAD_TOKEN)
    out = mlm(input_tensor, labels, training=True)
    assert out.plm_target_mapping is not None
    assert out.plm_perm_mask is not None
    
    
    