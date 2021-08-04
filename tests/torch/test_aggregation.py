import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_concat_aggregation_yoochoose(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group
    tab_module = torch4rec.features.tabular.TabularFeatures.from_column_group(col_group)

    block = tab_module >> torch4rec.ConcatFeatures()

    out = block(torch_yoochoose_like)

    assert out.shape[-1] == 248


def test_stack_aggregation_yoochoose(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group
    tab_module = torch4rec.EmbeddingFeatures.from_column_group(col_group)

    block = tab_module >> torch4rec.StackFeatures()

    out = block(torch_yoochoose_like)

    assert out.shape[1] == 64
    assert out.shape[2] == 2


def test_element_wise_sum_aggregation_yoochoose(yoochoose_column_group, torch_yoochoose_like):
    col_group = yoochoose_column_group
    tab_module = torch4rec.EmbeddingFeatures.from_column_group(col_group)

    block = tab_module >> torch4rec.ElementwiseSum()

    out = block(torch_yoochoose_like)

    assert out.shape[-1] == 64
