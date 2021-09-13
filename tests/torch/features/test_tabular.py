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

torch4rec = pytest.importorskip("transformers4rec.torch")
torch_utils = pytest.importorskip("transformers4rec.torch.utils.torch_utils")


def test_tabular_features(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.TabularFeatures.from_schema(schema)

    outputs = tab_module(torch_yoochoose_like)

    assert set(outputs.keys()) == set(
        schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )


def test_tabular_features_embeddings_options(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema

    EMB_DIM = 100
    tab_module = torch4rec.TabularFeatures.from_schema(schema, embedding_dim_default=EMB_DIM)

    outputs = tab_module(torch_yoochoose_like)

    categ_features = schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(v.shape[-1] == EMB_DIM for k, v in outputs.items() if k in categ_features)


def test_tabular_features_with_projection(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.TabularFeatures.from_schema(
        schema, max_sequence_length=20, continuous_projection=64
    )

    outputs = tab_module(torch_yoochoose_like)

    assert len(outputs.keys()) == 3
    assert all(tensor.shape[-1] == 64 for tensor in outputs.values())


def test_tabular_features_soft_encoding(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema

    emb_cardinality = 10
    emb_dim = 8
    tab_module = torch4rec.TabularFeatures.from_schema(
        schema,
        continuous_soft_embeddings=True,
        soft_embedding_cardinality_default=emb_cardinality,
        soft_embedding_dim_default=emb_dim,
    )

    outputs = tab_module(torch_yoochoose_like)

    assert (
        list(outputs.keys())
        == schema.select_by_tag(Tag.CONTINUOUS).column_names
        + schema.select_by_tag(Tag.CATEGORICAL).column_names
    )

    assert all(
        list(outputs[col_name].shape) == list(torch_yoochoose_like[col_name].shape) + [emb_dim]
        for col_name in schema.select_by_tag(Tag.CONTINUOUS).column_names
    )
