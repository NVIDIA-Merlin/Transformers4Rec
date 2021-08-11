import pytest

from transformers4rec.utils.tags import Tag

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_continuous_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = torch4rec.ContinuousFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    cont_cols = schema.select_by_tag(Tag.CONTINUOUS)

    con = torch4rec.ContinuousFeatures.from_schema(cont_cols)
    outputs = con(torch_yoochoose_like)

    assert list(outputs.keys()) == cont_cols.column_names
