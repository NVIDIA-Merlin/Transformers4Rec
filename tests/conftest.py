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

import merlin_standard_lib as msl
from merlin_standard_lib import Schema, Tag
from transformers4rec.data import tabular_sequence_testing_data, tabular_testing_data


@pytest.fixture
def yoochoose_path_file() -> str:
    return tabular_sequence_testing_data.path


@pytest.fixture
def yoochoose_schema_file() -> str:
    return tabular_sequence_testing_data.schema_path


@pytest.fixture
def yoochoose_schema() -> Schema:
    return tabular_sequence_testing_data.schema


@pytest.fixture
def tabular_data_file() -> str:
    return tabular_testing_data.path


@pytest.fixture
def tabular_schema_file() -> str:
    return tabular_testing_data.schema_path


@pytest.fixture
def tabular_schema() -> Schema:
    return tabular_testing_data.schema.remove_by_name(["session_id", "session_start", "day_idx"])


@pytest.fixture
def synthetic_schema() -> Schema:
    schema = Schema(
        [
            msl.ColumnSchema.create_categorical("session_id", num_items=5000, tags=["session_id"]),
            msl.ColumnSchema.create_categorical(
                "item_id",
                num_items=10000,
                tags=[Tag.ITEM_ID, Tag.LIST],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_categorical(
                "category",
                num_items=100,
                tags=[Tag.LIST, Tag.ITEM],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_continuous(
                "item_recency",
                min_value=0,
                max_value=1,
                tags=[Tag.LIST, Tag.ITEM],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_categorical("day", num_items=11, tags=[Tag.SESSION]),
            msl.ColumnSchema.create_categorical(
                "purchase", num_items=3, tags=[Tag.SESSION, Tag.BINARY_CLASSIFICATION]
            ),
            msl.ColumnSchema.create_continuous(
                "price", min_value=0, max_value=1, tags=[Tag.SESSION, Tag.REGRESSION]
            ),
        ]
    )

    return schema


from tests.tf.conftest import *  # noqa
from tests.torch.conftest import *  # noqa
