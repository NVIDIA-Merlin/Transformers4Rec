import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_continuous_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = torch4rec.ContinuousFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group

    con = torch4rec.ContinuousFeatures.from_column_group(col_group.continuous_column_group())
    outputs = con(torch_yoochoose_like)

    assert list(outputs.keys()) == col_group.continuous_columns()
