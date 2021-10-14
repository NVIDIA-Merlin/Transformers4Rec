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
from merlin_standard_lib.utils.embedding_utils import get_embedding_sizes_from_schema


def test_schema_from_yoochoose_schema(yoochoose_schema):
    assert len(yoochoose_schema.column_names) == 22
    assert len(yoochoose_schema.select_by_tag(Tag.CONTINUOUS).column_schemas) == 11
    assert len(yoochoose_schema.select_by_tag(Tag.CATEGORICAL).column_schemas) == 3


def test_schema_cardinalities(yoochoose_schema):
    schema = yoochoose_schema
    assert schema.categorical_cardinalities() == {
        "item_id/list": schema.select_by_name("item_id/list").feature[0].int_domain.max + 1,
        "category/list": schema.select_by_name("category/list").feature[0].int_domain.max + 1,
        "user_country": schema.select_by_name("user_country").feature[0].int_domain.max + 1,
    }


@pytest.mark.skip(reason="broken")
def test_schema_embedding_sizes_nvt(yoochoose_schema):
    pytest.importorskip("nvtabular")
    schema = yoochoose_schema
    assert schema.categorical_cardinalities() == {"item_id/list": 51996, "category/list": 332}
    embedding_sizes = schema.embedding_sizes_nvt(minimum_size=16, maximum_size=512)
    assert embedding_sizes == {"item_id/list": 512, "category/list": 41, "user_country": 16}


def test_schema_embedding_sizes(yoochoose_schema):
    schema = yoochoose_schema.remove_by_name("session_id")

    assert schema.categorical_cardinalities() == {
        "category/list": 333,
        "item_id/list": 51997,
        "user_country": 63,
    }
    embedding_sizes = get_embedding_sizes_from_schema(schema, multiplier=3.0)
    assert embedding_sizes == {"item_id/list": 46, "category/list": 13, "user_country": 9}
