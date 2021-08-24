import pytest

from transformers4rec.utils.tags import Tag

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_sequential_embedding_features(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = tf4rec.SequentialEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(tf_yoochoose_like)

    assert list(outputs.keys()) == schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(len(tensor.shape) == 3 for tensor in list(outputs.values()))
    assert all(tensor.shape[1] == 20 for tensor in list(outputs.values()))
    assert all(tensor.shape[2] == 64 for tensor in list(outputs.values()))


def test_sequential_tabular_features_with_projection(yoochoose_schema, tf_yoochoose_like):
    tab_module = tf4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, continuous_projection=64
    )

    outputs = tab_module(tf_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
    assert all(tensor.shape[1] == 20 for tensor in outputs.values())


# Add these tests when we port Masking to TF

# def test_sequential_tabular_features_with_masking(yoochoose_schema, tf_yoochoose_like):
#     input_module = tf4rec.SequentialTabularFeatures.from_schema(
#         yoochoose_schema,
#         max_sequence_length=20,
#         continuous_projection=64,
#         d_output=100,
#         masking="causal",
#     )
#
#     outputs = input_module(tf_yoochoose_like)
#
#     assert outputs.ndim == 3
#     assert outputs.shape[-1] == 100
#     assert outputs.shape[1] == 20
#
#
# def test_sequential_tabular_features_with_masking_no_itemid(yoochoose_schema):
#     with pytest.raises(ValueError) as excinfo:
#
#         yoochoose_schema = yoochoose_schema - ["item_id/list"]
#
#         tf4rec.SequentialTabularFeatures.from_schema(
#             yoochoose_schema,
#             max_sequence_length=20,
#             continuous_projection=64,
#             d_output=100,
#             masking="causal",
#         )
#
#     err = excinfo.value
#     assert "For masking a categorical_module is required including an item_id" in str(err)


def test_sequential_tabular_features_with_projection_and_d_output(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:

        tf4rec.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            projection=tf4rec.MLPBlock([64]),
            masking="causal",
        )

    assert "You cannot specify both d_output and projection at the same time" in str(excinfo.value)
