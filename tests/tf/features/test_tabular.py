import pytest

from transformers4rec.utils.tags import Tag

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_tabular_features(yoochoose_schema, tf_yoochoose_like):
    tab_module = tf4rec.TabularFeatures.from_schema(yoochoose_schema)

    outputs = tab_module(tf_yoochoose_like)

    assert (
        list(outputs.keys())
        == yoochoose_schema.select_by_tag(Tag.CONTINUOUS).column_names
        + yoochoose_schema.select_by_tag(Tag.CATEGORICAL).column_names
    )


def test_tabular_features_with_projection(yoochoose_schema, tf_yoochoose_like):
    tab_module = tf4rec.TabularFeatures.from_schema(yoochoose_schema, continuous_projection=64)

    outputs = tab_module(tf_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(len(tensor.shape) == 2 for tensor in outputs.values())
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
