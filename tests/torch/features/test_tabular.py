import pytest

from transformers4rec.utils.tags import Tag

torch4rec = pytest.importorskip("transformers4rec.torch")
torch_utils = pytest.importorskip("transformers4rec.torch.utils.torch_utils")


def test_tabular_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.TabularFeatures.from_schema(schema)

    outputs = tab_module(torch_yoochoose_like)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )


def test_tabular_features_with_projection(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.TabularFeatures.from_schema(
        schema, max_sequence_length=20, continuous_projection=64
    )

    outputs = tab_module(torch_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
