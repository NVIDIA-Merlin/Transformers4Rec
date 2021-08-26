from functools import partial

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


def test_embedding_features_custom_init(torch_cat_features):
    MEAN = 1.0
    STD = 0.05
    emb_initializer = partial(pytorch.nn.init.normal_, mean=MEAN, std=STD)
    feature_config = {
        f: torch4rec.FeatureConfig(
            torch4rec.TableConfig(100, dim=15, name=f, initializer=emb_initializer)
        )
        for f in torch_cat_features.keys()
    }
    embeddings = torch4rec.EmbeddingFeatures(feature_config, layer_norm=False)(torch_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all(
        [emb.detach().numpy().mean() == pytest.approx(MEAN, abs=0.1) for emb in embeddings.values()]
    )
    assert all(
        [emb.detach().numpy().std() == pytest.approx(STD, abs=0.1) for emb in embeddings.values()]
    )


def test_table_config_invalid_embedding_initializer():
    with pytest.raises(ValueError) as excinfo:
        torch4rec.TableConfig(100, dim=15, initializer="INVALID INITIALIZER")
    assert "initializer must be callable if specified" in str(excinfo.value)


def test_embedding_features_yoochoose(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = torch4rec.EmbeddingFeatures.from_schema(schema)
    embeddings = emb_module(torch_yoochoose_like)

    assert list(embeddings.keys()) == schema.column_names
    assert all(emb.shape[-1] == 64 for emb in embeddings.values())
    assert emb_module.item_id == "item_id/list"
    assert emb_module.item_embedding_table.num_embeddings == 51996


def test_embedding_features_yoochoose_custom_dims(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = torch4rec.EmbeddingFeatures.from_schema(
        schema, embedding_dims={"item_id/list": 100}, default_embedding_dim=64
    )

    assert emb_module.embedding_tables["item_id/list"].weight.shape[1] == 100
    assert emb_module.embedding_tables["category/list"].weight.shape[1] == 64

    embeddings = emb_module(torch_yoochoose_like)

    assert embeddings["item_id/list"].shape[1] == 100
    assert embeddings["category/list"].shape[1] == 64


def test_embedding_features_yoochoose_infer_embedding_sizes(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = torch4rec.EmbeddingFeatures.from_schema(
        schema, infer_embedding_sizes=True, infer_embedding_sizes_multiplier=3.0
    )

    assert emb_module.embedding_tables["item_id/list"].weight.shape[1] == 46
    assert emb_module.embedding_tables["category/list"].weight.shape[1] == 13

    embeddings = emb_module(torch_yoochoose_like)

    assert embeddings["item_id/list"].shape[1] == 46
    assert embeddings["category/list"].shape[1] == 13


def test_embedding_features_yoochoose_custom_initializers(yoochoose_schema, torch_yoochoose_like):
    ITEM_MEAN = 1.0
    ITEM_STD = 0.05

    CATEGORY_MEAN = 2.0
    CATEGORY_STD = 0.1

    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = torch4rec.EmbeddingFeatures.from_schema(
        schema,
        layer_norm=False,
        embeddings_initializers={
            "item_id/list": partial(pytorch.nn.init.normal_, mean=ITEM_MEAN, std=ITEM_STD),
            "category/list": partial(pytorch.nn.init.normal_, mean=CATEGORY_MEAN, std=CATEGORY_STD),
        },
    )

    embeddings = emb_module(torch_yoochoose_like)

    assert embeddings["item_id/list"].detach().numpy().mean() == pytest.approx(ITEM_MEAN, abs=0.1)
    assert embeddings["item_id/list"].detach().numpy().std() == pytest.approx(ITEM_STD, abs=0.1)

    assert embeddings["category/list"].detach().numpy().mean() == pytest.approx(
        CATEGORY_MEAN, abs=0.1
    )
    assert embeddings["category/list"].detach().numpy().std() == pytest.approx(
        CATEGORY_STD, abs=0.1
    )


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
