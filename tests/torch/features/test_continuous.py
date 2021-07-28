import pytest

from transformers4rec.utils.columns import ColumnGroup

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_continuous_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = torch4rec.ContinuousFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_schema_file, torch_yoochoose_like):
    col_group = ColumnGroup.from_schema(str(yoochoose_schema_file))

    con = torch4rec.ContinuousFeatures.from_column_group(col_group.continuous_column_group())
    outputs = con(torch_yoochoose_like)

    assert list(outputs.keys()) == col_group.continuous_columns()
