import pytest

from ..conftest import MAX_CARDINALITY

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_embedding_features(torch_cat_features):
    dim = 15
    feature_config = {
        f: torch4rec.FeatureConfig(torch4rec.TableConfig(MAX_CARDINALITY, dim, name=f))
        for f in torch_cat_features.keys()
    }
    embeddings = torch4rec.EmbeddingFeatures(feature_config)(torch_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


def test_soft_continuous_features(torch_con_features):
    dim = 16
    num_embeddings = 64
    feature_config = {
        f: torch4rec.FeatureConfig(torch4rec.TableConfig(num_embeddings, dim, name=f))
        for f in torch_con_features.keys()
    }
    con_embeddings = torch4rec.SoftEmbeddingFeatures(feature_config)(torch_con_features)

    assert list(con_embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in con_embeddings.values()])
