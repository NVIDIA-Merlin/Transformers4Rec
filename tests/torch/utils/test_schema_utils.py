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

from merlin_standard_lib import ColumnSchema, Schema, Tag
from merlin_standard_lib.schema.schema import ValueCount

schema_utils = pytest.importorskip("transformers4rec.torch.utils.schema_utils")
pytorch = pytest.importorskip("torch")


def test_random_data_from_simple_schema():
    s = Schema(
        [
            ColumnSchema.create_categorical(
                "item_id",
                num_items=1000,
                tags=[Tag.ITEM_ID, Tag.LIST],
                value_count=ValueCount(1, 50),
            ),
            ColumnSchema.create_categorical(
                "session_cat", num_items=1000, tags=[Tag.LIST], value_count=ValueCount(1, 50)
            ),
            ColumnSchema.create_continuous(
                "session_con", tags=[Tag.LIST], value_count=ValueCount(1, 50)
            ),
            ColumnSchema.create_categorical("context_cat", num_items=1000),
        ]
    )

    random_data = schema_utils.random_data_from_schema(s, 100, max_session_length=50)

    assert random_data["context_cat"].shape == (100,)
    assert random_data["session_con"].dtype == pytorch.float32
    for val in s.select_by_tag(Tag.LIST).filter_columns_from_dict(random_data).values():
        assert val.shape == (100, 50)

    for val in s.select_by_tag(Tag.CATEGORICAL).filter_columns_from_dict(random_data).values():
        assert val.dtype == pytorch.int64
        assert val.max() < 1000


def test_random_data_from_schema_with_embeddings():
    def create_emb(name):
        return ColumnSchema.create_continuous(name, shape=(100,), is_embedding=True)

    s = Schema([create_emb("item-embedding"), create_emb("user-embedding")])

    random_data = schema_utils.random_data_from_schema(s, 100)

    for val in random_data.values():
        assert val.shape == (100, 100)
