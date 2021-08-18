import pytest

from transformers4rec.utils.tags import Tag

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_embedding_features(torch_cat_features):
    dim = 15
    feature_config = {
        f: torch4rec.FeatureConfig(torch4rec.TableConfig(100, dim, name=f))
        for f in torch_cat_features.keys()
    }
    embeddings = torch4rec.EmbeddingFeatures(feature_config)(torch_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


def test_embedding_features_yoochoose(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = torch4rec.EmbeddingFeatures.from_schema(schema)
    embeddings = emb_module(torch_yoochoose_like)

    assert list(embeddings.keys()) == schema.column_names
    assert all(emb.shape[-1] == 64 for emb in embeddings.values())
    assert emb_module.item_id == "item_id/list"
    assert emb_module.item_embedding_table.num_embeddings == 51996


def test_soft_embedding_invalid_num_embeddings():
    with pytest.raises(AssertionError) as excinfo:
        torch4rec.SoftEmbedding(num_embeddings=0, embeddings_dim=16)
    assert "number of embeddings for soft embeddings needs to be greater than 0" in str(
        excinfo.value
    )


def test_soft_embedding_invalid_embeddings_dim():
    with pytest.raises(AssertionError) as excinfo:
        torch4rec.SoftEmbedding(num_embeddings=10, embeddings_dim=0)
    assert "embeddings dim for soft embeddings needs to be greater than 0" in str(excinfo.value)


def test_soft_embedding():
    embeddings_dim = 16
    num_embeddings = 64

    soft_embedding = torch4rec.SoftEmbedding(
        num_embeddings, embeddings_dim, embeddings_init_std=0.05
    )
    assert soft_embedding.embedding_table.weight.shape == pytorch.Size(
        [num_embeddings, embeddings_dim]
    ), "Internal soft embedding table does not have the expected shape"

    batch_size = 10
    seq_length = 20
    cont_feature_inputs = pytorch.rand((batch_size, seq_length))
    output = soft_embedding(cont_feature_inputs)

    assert output.shape == pytorch.Size(
        [batch_size, seq_length, embeddings_dim]
    ), "Soft embedding output has not the expected shape"


def test_soft_continuous_features(torch_con_features):
    dim = 16
    num_embeddings = 64
    feature_config = {
        f: torch4rec.FeatureConfig(torch4rec.TableConfig(num_embeddings, dim, name=f))
        for f in torch_con_features.keys()
    }
    con_embeddings = torch4rec.SoftEmbeddingFeatures(feature_config, soft_embeddings_init_std=0.05)(
        torch_con_features
    )

    assert list(con_embeddings.keys()) == list(feature_config.keys())
    assert all(
        [
            list(v.shape) == list(torch_con_features[k].shape) + [dim]
            for k, v in con_embeddings.items()
        ]
    )
