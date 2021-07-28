import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_tabular_features(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group
    tab_module = torch4rec.features.tabular.TabularFeatures.from_column_group(col_group)

    outputs = tab_module(torch_yoochoose_like)

    assert list(outputs.keys()) == col_group.continuous_columns() + col_group.categorical_columns()
