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

from merlin_standard_lib import Tag

tr = pytest.importorskip("transformers4rec.torch")


def test_sequential_and_non_seq_embedding_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = tr.SequenceEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(torch_yoochoose_like)

    assert list(outputs.keys()) == schema.select_by_tag(Tag.CATEGORICAL).column_names

    seq_features = ["item_id/list", "category/list"]
    non_seq_features = ["user_country"]

    for fname in seq_features:
        assert list(outputs[fname].shape) == [100, 20, 64]

    for fname in non_seq_features:
        assert list(outputs[fname].shape) == [100, 64]


def test_sequential_tabular_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tr.TabularSequenceFeatures.from_schema(schema)

    outputs = tab_module(torch_yoochoose_like)

    tag_select = lambda tags: any(t in [Tag.CONTINUOUS, Tag.CATEGORICAL] for t in tags)  # noqa
    cols = schema.select_by_tag(tag_select).column_names

    assert set(outputs.keys()) == set(cols)


def test_sequential_tabular_features_with_feature_modules_kwargs(
    yoochoose_schema, torch_yoochoose_like
):
    schema = yoochoose_schema
    EMB_DIM = 200
    tab_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        embedding_dim_default=EMB_DIM,
    )

    outputs = tab_module(torch_yoochoose_like)

    assert set(outputs.keys()) == set(
        schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )

    categ_features = schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(v.shape[-1] == EMB_DIM for k, v in outputs.items() if k in categ_features)


def test_sequential_tabular_features_with_projection(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tr.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=20, continuous_projection=64
    )
    continuous_feature_names = schema.select_by_tag(Tag.CONTINUOUS).column_names

    outputs = tab_module(torch_yoochoose_like)

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1:] == [20, 64]


def test_sequential_tabular_features_with_masking(yoochoose_schema, torch_yoochoose_like):
    input_module = tr.TabularSequenceFeatures.from_schema(
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


def test_tabular_features_yoochoose_direct(yoochoose_schema, torch_yoochoose_like):
    continuous_module = tr.ContinuousFeatures.from_schema(yoochoose_schema, tags=["continuous"])
    categorical_module = tr.SequenceEmbeddingFeatures.from_schema(
        yoochoose_schema, tags=["categorical"]
    )

    tab_seq_features = tr.TabularSequenceFeatures(
        continuous_module=continuous_module,
        categorical_module=categorical_module,
        aggregation="concat",
        schema=yoochoose_schema,
    )
    outputs = tab_seq_features(torch_yoochoose_like)

    assert (
        len(
            set(categorical_module.schema.column_names).difference(
                set(tab_seq_features.schema.column_names)
            )
        )
        == 0
    )
    assert (
        len(
            set(continuous_module.schema.column_names).difference(
                set(tab_seq_features.schema.column_names)
            )
        )
        == 0
    )
    assert len(outputs.shape) == 3


def test_sequential_tabular_features_with_masking_no_itemid(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        yoochoose_schema = yoochoose_schema.remove_by_name("item_id/list")

        tr.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            masking="causal",
        )

    assert "For masking a categorical_module is required including an item_id" in str(excinfo.value)


def test_sequential_tabular_features_with_projection_and_d_output(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        tr.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            projection=tr.MLPBlock([64]),
            masking="causal",
        )

    assert "You cannot specify both d_output and projection at the same time" in str(excinfo.value)


def test_sequential_and_non_sequential_tabular_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tr.TabularSequenceFeatures.from_schema(schema, aggregation="concat")

    outputs = tab_module(torch_yoochoose_like)

    assert list(outputs.shape) == [100, 20, 203]
