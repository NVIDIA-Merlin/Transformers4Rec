import pytest

NUM_EXAMPLES = 1000


@pytest.fixture
def tf_con_features():
    tf = pytest.importorskip("tensorflow")

    features = {}
    keys = "abcdef"

    for key in keys:
        features[key] = tf.random.uniform((NUM_EXAMPLES, 1))

    return features
