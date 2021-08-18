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


def test_tabular_features_soft_encoding(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema

    emb_cardinality = 10
    emb_dim = 8
    tab_module = torch4rec.TabularFeatures.from_schema(
        schema, continuous_soft_embeddings_shape=(emb_cardinality, emb_dim)
    )

    outputs = tab_module(torch_yoochoose_like)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )

    assert all(
        list(outputs[col_name].shape) == list(torch_yoochoose_like[col_name].shape) + [emb_dim]
        for col_name in schema.select_by_tag(Tag.CONTINUOUS).column_names
    )


def test_tabular_features_soft_encoding_invalid_shape(yoochoose_schema):
    with pytest.raises(AssertionError) as excinfo:
        torch4rec.TabularFeatures.from_schema(
            yoochoose_schema, continuous_soft_embeddings_shape=(10)
        )
    assert "continuous_soft_embeddings_shape must be a list/tuple with 2 elements" in str(
        excinfo.value
    )
