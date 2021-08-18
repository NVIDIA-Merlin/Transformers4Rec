import pytest

from transformers4rec.utils.tags import Tag

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_sequential_embedding_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    emb_module = torch4rec.SequentialEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(torch_yoochoose_like)

    assert list(outputs.keys()) == schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(tensor.shape[1] == 20 for tensor in list(outputs.values()))
    assert all(tensor.shape[2] == 64 for tensor in list(outputs.values()))


def test_sequential_tabular_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(schema)

    outputs = tab_module(torch_yoochoose_like)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )


def test_sequential_tabular_features_with_projection(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        schema, max_sequence_length=20, continuous_projection=64
    )

    outputs = tab_module(torch_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
    assert all(tensor.shape[1] == 20 for tensor in outputs.values())


def test_sequential_tabular_features_with_masking(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.SequentialTabularFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )

    outputs = input_module(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 100
    assert outputs.shape[1] == 20
