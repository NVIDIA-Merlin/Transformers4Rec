import pytest

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_continuous_features(tf_con_features):
    features = ["a", "b"]
    con = tf4rec.ContinuousFeatures(features)(tf_con_features)

    assert list(con.keys()) == features
