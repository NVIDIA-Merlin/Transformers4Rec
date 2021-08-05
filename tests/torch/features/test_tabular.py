import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")
torch_utils = pytest.importorskip("transformers4rec.torch.utils.torch_utils")


def test_tabular_features(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group
    tab_module = torch4rec.TabularFeatures.from_column_group(col_group)

    outputs = tab_module(torch_yoochoose_like)

    assert list(outputs.keys()) == col_group.continuous_columns() + col_group.categorical_columns()


def test_tabular_features_with_projection(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group
    tab_module = torch4rec.TabularFeatures.from_column_group(
        col_group, max_sequence_length=20, continuous_projection=64
    )

    outputs = tab_module(torch_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
