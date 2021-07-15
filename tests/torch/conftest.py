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
