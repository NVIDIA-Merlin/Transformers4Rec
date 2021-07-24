import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
nvt = pytest.importorskip("nvtabular")

NUM_EXAMPLES = 1000
MAX_CARDINALITY = 100


@pytest.fixture
def torch_con_features():
    features = {}
    keys = [f"con_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = torch.rand((NUM_EXAMPLES, 1))

    return features


@pytest.fixture
def torch_cat_features():

    features = {}
    keys = [f"cat_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = torch.randint(MAX_CARDINALITY, (NUM_EXAMPLES,))

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
    features["input_tensor"] = torch.tensor(
        np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim))
    )
    # create sequences
    labels = torch.tensor(np.random.randint(1, MAX_CARDINALITY, (NUM_EXAMPLES, MAX_LEN)))
    # replace last 2 items by zeros to mimic padding
    labels[:, MAX_LEN - 2 :] = 0
    features["labels"] = labels
    features["pad_token"] = PAD_TOKEN
    features["vocab_size"] = MAX_CARDINALITY

    return features


@pytest.fixture
def torch_seq_prediction_head_inputs():
    ITEM_DIM = 128
    POS_EXAMPLE = 25
    features = {}
    features["seq_model_output"] = torch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, ITEM_DIM)))
    features["item_embedding_table"] = torch.nn.Embedding(MAX_CARDINALITY, ITEM_DIM)
    features["labels_all"] = torch.tensor(np.random.randint(1, MAX_CARDINALITY, (POS_EXAMPLE,)))
    features["vocab_size"] = MAX_CARDINALITY
    features["item_dim"] = ITEM_DIM
    return features


@pytest.fixture
def torch_ranking_metrics_inputs():
    POS_EXAMPLE = 30
    VOCAB_SIZE = 40
    features = {}
    features["scores"] = torch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, VOCAB_SIZE)))
    features["ks"] = torch.LongTensor([1, 2, 3, 5, 10, 20])
    features["labels_one_hot"] = torch.LongTensor(
        np.random.choice(a=[0, 1], size=(POS_EXAMPLE, VOCAB_SIZE))
    )

    features["labels"] = torch.tensor(np.random.randint(1, VOCAB_SIZE, (POS_EXAMPLE,)))
    return features


@pytest.fixture
def torch_seq_prediction_head_link_to_block():
    ITEM_DIM = 64
    POS_EXAMPLE = 25
    features = {}
    features["seq_model_output"] = torch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, ITEM_DIM)))
    features["item_embedding_table"] = torch.nn.Embedding(MAX_CARDINALITY, ITEM_DIM)
    features["labels_all"] = torch.tensor(np.random.randint(1, MAX_CARDINALITY, (POS_EXAMPLE,)))
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
