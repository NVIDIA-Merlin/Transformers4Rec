import pytest

NUM_EXAMPLES = 1000


@pytest.fixture
def torch_con_features():
    torch = pytest.importorskip("torch")

    features = {}
    keys = "abcdef"

    for key in keys:
        features[key] = torch.rand((NUM_EXAMPLES, 1))

    return features


@pytest.fixture
def torch_masking_inputs():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    # fixed parameters for tests
    MAX_LEN = 10
    NUM_EXAMPLES = 8
    PAD_TOKEN = 0
    VOCAB_SIZE = 100
    hidden_dim = 16
    features = {}
    # generate random tensors for test
    features["input_tensor"] = torch.tensor(
        np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim))
    )
    # create sequences
    labels = torch.tensor(np.random.randint(1, VOCAB_SIZE, (NUM_EXAMPLES, MAX_LEN)))
    # replace last 2 items by zeros to mimic padding
    labels[:, MAX_LEN - 2 :] = 0
    features["labels"] = labels
    features["pad_token"] = PAD_TOKEN
    features["vocab_size"] = VOCAB_SIZE
    return features
