import pytest

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_continuous_features(torch_con_features):
    features = ["a", "b"]
    con = torch4rec.ContinuousFeatures(features)(torch_con_features)

    assert list(con.keys()) == features
