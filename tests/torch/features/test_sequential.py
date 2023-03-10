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
from merlin.schema import Schema as CoreSchema
from merlin.schema import Tags

import transformers4rec.torch as tr
from tests.conftest import parametrize_schemas


@parametrize_schemas("yoochoose")
def test_sequential_and_non_seq_embedding_features(schema, torch_yoochoose_like):
    schema = schema.select_by_tag(Tags.CATEGORICAL)
    emb_module = tr.SequenceEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(torch_yoochoose_like)

    assert list(outputs.keys()) == schema.select_by_tag(Tags.CATEGORICAL).column_names

    seq_features = ["item_id/list", "category/list"]
    non_seq_features = ["user_country"]

    for fname in seq_features:
        assert list(outputs[fname].shape) == [100, 20, 64]

    for fname in non_seq_features:
        assert list(outputs[fname].shape) == [100, 64]


@parametrize_schemas("yoochoose")
def test_sequential_tabular_features(schema, torch_yoochoose_like):
    tab_module = tr.TabularSequenceFeatures.from_schema(schema)

    outputs = tab_module(torch_yoochoose_like)

    cols = [
        c.name
        for c in list(
            schema.select_by_tag(Tags.CONTINUOUS) + schema.select_by_tag(Tags.CATEGORICAL)
        )
    ]

    assert set(outputs.keys()) == set(cols)


@parametrize_schemas("yoochoose")
def test_sequential_tabular_features_with_feature_modules_kwargs(schema, torch_yoochoose_like):
    EMB_DIM = 200
    tab_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        embedding_dim_default=EMB_DIM,
    )

    outputs = tab_module(torch_yoochoose_like)

    assert set(outputs.keys()) == set(
        schema.select_by_tag(Tags.CONTINUOUS).column_names
        + schema.select_by_tag(Tags.CATEGORICAL).column_names
    )

    categ_features = schema.select_by_tag(Tags.CATEGORICAL).column_names
    assert all(v.shape[-1] == EMB_DIM for k, v in outputs.items() if k in categ_features)


@parametrize_schemas("yoochoose")
def test_sequential_tabular_features_with_projection(schema, torch_yoochoose_like):
    tab_module = tr.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=20, continuous_projection=64
    )
    continuous_feature_names = schema.select_by_tag(Tags.CONTINUOUS).column_names

    outputs = tab_module(torch_yoochoose_like)

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1:] == [20, 64]


@parametrize_schemas("yoochoose")
def test_sequential_tabular_features_with_masking(schema, torch_yoochoose_like):
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )

    outputs = input_module(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 100
    assert outputs.shape[1] == 20


@parametrize_schemas("yoochoose")
def test_sequential_tabular_features_ignore_masking(schema, torch_yoochoose_like):
    import numpy as np

    from transformers4rec.torch.masking import CausalLanguageModeling, MaskedLanguageModeling

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        aggregation="concat",
    )
    output_wo_masking = input_module(torch_yoochoose_like, training=False).detach().cpu().numpy()

    input_module._masking = CausalLanguageModeling(hidden_size=100)

    output_inference_masking = (
        input_module(torch_yoochoose_like, training=False, testing=False).detach().cpu().numpy()
    )
    output_clm_masking = (
        input_module(torch_yoochoose_like, training=False, testing=True).detach().cpu().numpy()
    )

    assert np.allclose(output_wo_masking, output_inference_masking, rtol=1e-04, atol=1e-08)
    assert not np.allclose(output_wo_masking, output_clm_masking, rtol=1e-04, atol=1e-08)

    input_module._masking = MaskedLanguageModeling(hidden_size=100)
    output_inference_masking = (
        input_module(torch_yoochoose_like, training=False, testing=False).detach().cpu().numpy()
    )
    output_eval_masking = (
        input_module(torch_yoochoose_like, training=False, testing=True).detach().cpu().numpy()
    )
    # MLM extends the inputs with one position during inference
    assert output_inference_masking.shape[1] == output_eval_masking.shape[1] + 1


@parametrize_schemas("yoochoose")
def test_tabular_features_yoochoose_direct(schema, torch_yoochoose_like):
    continuous_module = tr.ContinuousFeatures.from_schema(schema, tags=Tags.CONTINUOUS)
    categorical_module = tr.SequenceEmbeddingFeatures.from_schema(schema, tags=Tags.CATEGORICAL)

    tab_seq_features = tr.TabularSequenceFeatures(
        continuous_module=continuous_module,
        categorical_module=categorical_module,
        aggregation="concat",
        schema=schema,
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


@parametrize_schemas("yoochoose")
def test_sequential_tabular_features_with_masking_no_itemid(schema):
    with pytest.raises(ValueError) as excinfo:
        if isinstance(schema, CoreSchema):
            schema = schema.excluding_by_name("item_id/list")
        else:
            schema = schema.remove_by_name("item_id/list")

        tr.TabularSequenceFeatures.from_schema(
            schema,
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


@parametrize_schemas("yoochoose")
def test_sequential_and_non_sequential_tabular_features(schema, torch_yoochoose_like):
    tab_module = tr.TabularSequenceFeatures.from_schema(schema, aggregation="concat")

    outputs = tab_module(torch_yoochoose_like)

    assert list(outputs.shape) == [100, 20, 203]
