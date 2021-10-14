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

tr = pytest.importorskip("transformers4rec.tf")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")


def test_sequence_embedding_features(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = tr.SequenceEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(tf_yoochoose_like)

    categ_schema = schema.select_by_tag(Tag.CATEGORICAL)
    assert list(outputs.keys()) == categ_schema.column_names

    sequential_categ_cols = categ_schema.select_by_tag(Tag.LIST).column_names

    for k in categ_schema.column_names:
        tensor = outputs[k]
        if k in sequential_categ_cols:
            assert len(tensor.shape) == 3
            assert tensor.shape[1] == 20
            assert tensor.shape[2] == 64
        else:
            assert len(tensor.shape) == 2
            assert tensor.shape[1] == 64


def test_serialization_sequence_embedding_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tr.SequenceEmbeddingFeatures.from_schema(yoochoose_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.feature_config.keys()) == list(copy_layer.feature_config.keys())

    from transformers4rec.tf.features.embedding import serialize_table_config as ser

    assert all(
        ser(inputs.feature_config[key].table) == ser(copy_layer.feature_config[key].table)
        for key in copy_layer.feature_config
    )


@test_utils.mark_run_eagerly_modes
def test_sequence_embedding_features_yoochoose_model(
    yoochoose_schema, tf_yoochoose_like, run_eagerly
):
    inputs = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    body = tr.SequentialBlock([inputs, tr.MLPBlock([64])])

    test_utils.assert_body_works_in_model(tf_yoochoose_like, inputs, body, run_eagerly)


def test_sequence_tabular_features_with_projection(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, continuous_projection=64
    )

    continuous_feature_names = schema.select_by_tag(Tag.CONTINUOUS).column_names

    outputs = tab_module(tf_yoochoose_like)

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1:] == [20, 64]


def test_serialization_sequence_tabular_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tr.TabularSequenceFeatures.from_schema(yoochoose_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.to_merge.keys()) == list(copy_layer.to_merge.keys())


@test_utils.mark_run_eagerly_modes
def test_tabular_features_yoochoose_direct(
    yoochoose_schema,
    tf_yoochoose_like,
    run_eagerly,
):
    continuous_layer = tr.ContinuousFeatures.from_schema(yoochoose_schema, tags=["continuous"])
    categorical_layer = tr.SequenceEmbeddingFeatures.from_schema(
        yoochoose_schema, tags=["categorical"]
    )

    tab_seq_features = tr.TabularSequenceFeatures(
        continuous_layer=continuous_layer,
        categorical_layer=categorical_layer,
        aggregation="concat",
    )
    outputs = tab_seq_features(tf_yoochoose_like)

    assert (
        len(
            set(categorical_layer.schema.column_names).difference(
                set(tab_seq_features.schema.column_names)
            )
        )
        == 0
    )
    assert (
        len(
            set(continuous_layer.schema.column_names).difference(
                set(tab_seq_features.schema.column_names)
            )
        )
        == 0
    )
    assert len(outputs.shape) == 3


def test_sequential_tabular_features_with_masking(yoochoose_schema, tf_yoochoose_like):
    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )

    outputs = input_module(tf_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 100
    assert outputs.shape[1] == 20


def test_sequential_tabular_features_with_masking_no_itemid(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:

        yoochoose_schema = yoochoose_schema.remove_by_name(["item_id/list"])

        tr.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            masking="causal",
        )

    err = excinfo.value
    assert "For masking a categorical_module is required including an item_id" in str(err)


def test_sequence_tabular_features_with_projection_and_d_output(yoochoose_schema):
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


def test_sequential_and_non_sequential_tabular_features(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tr.TabularSequenceFeatures.from_schema(schema, aggregation="concat")

    outputs = tab_module(tf_yoochoose_like)

    assert list(outputs.shape) == [100, 20, 203]
