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

from transformers4rec.utils.tags import Tag

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_sequential_embedding_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = torch4rec.SequenceEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(torch_yoochoose_like)

    assert list(outputs.keys()) == schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(len(tensor.shape) == 3 for tensor in list(outputs.values()))
    assert all(tensor.shape[1] == 20 for tensor in list(outputs.values()))
    assert all(tensor.shape[2] == 64 for tensor in list(outputs.values()))


def test_sequential_tabular_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.TabularSequenceFeatures.from_schema(schema)

    outputs = tab_module(torch_yoochoose_like)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )


def test_sequential_tabular_features_with_feature_modules_kwargs(
    yoochoose_schema, torch_yoochoose_like
):
    schema = yoochoose_schema
    EMB_DIM = 200
    tab_module = torch4rec.TabularSequenceFeatures.from_schema(
        schema,
        embedding_dim_default=EMB_DIM,
    )

    outputs = tab_module(torch_yoochoose_like)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )

    categ_features = schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(v.shape[-1] == EMB_DIM for k, v in outputs.items() if k in categ_features)


def test_sequential_tabular_features_with_projection(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=20, continuous_projection=64
    )

    outputs = tab_module(torch_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
    assert all(tensor.shape[1] == 20 for tensor in outputs.values())


def test_sequential_tabular_features_with_masking(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.TabularSequenceFeatures.from_schema(
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
    continuous_module = torch4rec.ContinuousFeatures.from_schema(
        yoochoose_schema, tags=["continuous"]
    )
    categorical_module = torch4rec.SequenceEmbeddingFeatures.from_schema(
        yoochoose_schema, tags=["categorical"]
    )

    inputs = torch4rec.TabularSequenceFeatures(
        continuous_module=continuous_module,
        categorical_module=categorical_module,
        aggregation="sequential-concat",
    )
    outputs = inputs(torch_yoochoose_like)

    assert inputs.schema == continuous_module.schema + categorical_module.schema
    assert len(outputs.shape) == 3


def test_sequential_tabular_features_with_masking_no_itemid(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        yoochoose_schema = yoochoose_schema - ["item_id/list"]

        torch4rec.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            masking="causal",
        )

    assert "For masking a categorical_module is required including an item_id" in str(excinfo.value)


def test_sequential_tabular_features_with_projection_and_d_output(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        torch4rec.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            projection=torch4rec.MLPBlock([64]),
            masking="causal",
        )

    assert "You cannot specify both d_output and projection at the same time" in str(excinfo.value)
