import pytest

from transformers4rec import tf as tf4rec

tf = pytest.importorskip("tensorflow")


# tf4rec = pytest.importorskip("transformers4rec.tf")


def test_continuous_features(continuous_features):
    features = ["scalar_continuous"]
    con = tf4rec.ContinuousFeatures(features)(continuous_features)

    assert list(con.keys()) == features
