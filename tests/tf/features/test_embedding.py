#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
from tensorflow.python.ops import init_ops_v2

from merlin_standard_lib import Tag

tr = pytest.importorskip("transformers4rec.tf")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")


def test_embedding_features(tf_cat_features):
    dim = 15
    feature_config = {
        f: tr.FeatureConfig(tr.TableConfig(100, dim, name=f, initializer=None))
        for f in tf_cat_features.keys()
    }
    embeddings = tr.EmbeddingFeatures(feature_config)(tf_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


def test_embedding_features_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = tr.EmbeddingFeatures.from_schema(schema)
    embeddings = emb_module(tf_yoochoose_like)

    assert list(embeddings.keys()) == schema.column_names
    assert all(emb.shape[-1] == 64 for emb in embeddings.values())
    assert emb_module.item_id == "item_id/list"
    max_value = schema.select_by_name("item_id/list").feature[0].int_domain.max
    assert emb_module.item_embedding_table.shape[0] == max_value + 1


def test_serialization_embedding_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tr.EmbeddingFeatures.from_schema(yoochoose_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.feature_config.keys()) == list(copy_layer.feature_config.keys())

    from transformers4rec.tf.features.embedding import serialize_table_config as ser

    assert all(
        ser(inputs.feature_config[key].table) == ser(copy_layer.feature_config[key].table)
        for key in copy_layer.feature_config
    )


@test_utils.mark_run_eagerly_modes
def test_embedding_features_yoochoose_model(yoochoose_schema, tf_yoochoose_like, run_eagerly):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    inputs = tr.EmbeddingFeatures.from_schema(schema, aggregation="concat")
    body = tr.SequentialBlock([inputs, tr.MLPBlock([64])])

    test_utils.assert_body_works_in_model(tf_yoochoose_like, inputs, body, run_eagerly)


def test_embedding_features_yoochoose_custom_dims(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = tr.EmbeddingFeatures.from_schema(
        schema, embedding_dims={"item_id/list": 100}, embedding_dim_default=64
    )

    embeddings = emb_module(tf_yoochoose_like)

    assert emb_module.embedding_tables["item_id/list"].shape[1] == 100
    assert emb_module.embedding_tables["category/list"].shape[1] == 64

    assert embeddings["item_id/list"].shape[1] == 100
    assert embeddings["category/list"].shape[1] == 64


def test_embedding_features_yoochoose_infer_embedding_sizes(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)

    emb_module = tr.EmbeddingFeatures.from_schema(
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
    emb_module = tr.EmbeddingFeatures.from_schema(
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
