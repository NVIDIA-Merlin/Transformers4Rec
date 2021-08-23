import pytest

from transformers4rec.utils.tags import Tag

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_continuous_features(tf_con_features):
    features = ["a", "b"]
    con = tf4rec.ContinuousFeatures(features)(tf_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    cont_cols = schema.select_by_tag(Tag.CONTINUOUS)

    con = tf4rec.ContinuousFeatures.from_schema(cont_cols)
    outputs = con(tf_yoochoose_like)

    assert list(outputs.keys()) == cont_cols.column_names
