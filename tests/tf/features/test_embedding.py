import pytest

from transformers4rec.utils.tags import Tag

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_embedding_features(tf_cat_features):
    dim = 15
    feature_config = {
        f: tf4rec.FeatureConfig(tf4rec.TableConfig(100, dim, name=f, initializer=None))
        for f in tf_cat_features.keys()
    }
    embeddings = tf4rec.EmbeddingFeatures(feature_config)(tf_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


def test_embedding_features_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = tf4rec.EmbeddingFeatures.from_schema(schema)
    embeddings = emb_module(tf_yoochoose_like)

    assert list(embeddings.keys()) == schema.column_names
    assert all(emb.shape[-1] == 64 for emb in embeddings.values())
    assert emb_module.item_id == "item_id/list"
    assert emb_module.item_embedding_table.shape[0] == 51996
