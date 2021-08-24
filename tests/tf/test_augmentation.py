import pytest

tf = pytest.importorskip("tensorflow")
tf4rec = pytest.importorskip("transformers4rec.tf")


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise(yoochoose_schema, tf_yoochoose_like, replacement_prob):
    tab_module = tf4rec.EmbeddingFeatures.from_schema(yoochoose_schema)

    block = tab_module >> tf4rec.StochasticSwapNoise(pad_token=0, replacement_prob=replacement_prob)
    out = (
        block(tf_yoochoose_like, training=True)["item_id/list"]
        == tab_module(tf_yoochoose_like)["item_id/list"]
    )
    replacement_rate = tf.reduce_mean(tf.cast(out, dtype=tf.float32)).numpy()

    assert replacement_rate == pytest.approx(1 - replacement_prob, 0.1)
