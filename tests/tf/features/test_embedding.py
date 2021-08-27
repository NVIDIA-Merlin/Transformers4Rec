import pytest
from tensorflow.python.ops import init_ops_v2

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


def test_embedding_features_yoochoose_custom_dims(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = tf4rec.EmbeddingFeatures.from_schema(
        schema, embedding_dims={"item_id/list": 100}, embedding_dim_default=64
    )

    embeddings = emb_module(tf_yoochoose_like)

    assert emb_module.embedding_tables["item_id/list"].shape[1] == 100
    assert emb_module.embedding_tables["category/list"].shape[1] == 64

    assert embeddings["item_id/list"].shape[1] == 100
    assert embeddings["category/list"].shape[1] == 64


def test_embedding_features_yoochoose_infer_embedding_sizes(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = tf4rec.EmbeddingFeatures.from_schema(
        schema, infer_embedding_sizes=True, infer_embedding_sizes_multiplier=3.0
    )

    embeddings = emb_module(tf_yoochoose_like)

    assert emb_module.embedding_tables["item_id/list"].shape[1] == 46
    assert emb_module.embedding_tables["category/list"].shape[1] == 13

    assert embeddings["item_id/list"].shape[1] == 46
    assert embeddings["category/list"].shape[1] == 13


def test_embedding_features_yoochoose_custom_initializers(yoochoose_schema, tf_yoochoose_like):
    ITEM_MEAN = 1.0
    ITEM_STD = 0.05

    CATEGORY_MEAN = 2.0
    CATEGORY_STD = 0.1

    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = tf4rec.EmbeddingFeatures.from_schema(
        schema,
        embeddings_initializers={
            "item_id/list": init_ops_v2.TruncatedNormal(mean=ITEM_MEAN, stddev=ITEM_STD),
            "category/list": init_ops_v2.TruncatedNormal(mean=CATEGORY_MEAN, stddev=CATEGORY_STD),
        },
    )

    embeddings = emb_module(tf_yoochoose_like)

    assert embeddings["item_id/list"].numpy().mean() == pytest.approx(ITEM_MEAN, abs=0.1)
    assert embeddings["item_id/list"].numpy().std() == pytest.approx(ITEM_STD, abs=0.1)

    assert embeddings["category/list"].numpy().mean() == pytest.approx(CATEGORY_MEAN, abs=0.1)
    assert embeddings["category/list"].numpy().std() == pytest.approx(CATEGORY_STD, abs=0.1)
