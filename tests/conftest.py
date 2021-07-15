import pytest

NUM_EXAMPLES = 1000
VECTOR_DIM = 128


@pytest.fixture
def continuous_features():
    tf = pytest.importorskip("tensorflow")

    scalar_feature = tf.random.uniform((NUM_EXAMPLES, 1))
    vector_feature = tf.random.uniform((NUM_EXAMPLES, VECTOR_DIM))

    return {
        "scalar_continuous": scalar_feature,
        "vector_continuous__values": vector_feature,
    }
