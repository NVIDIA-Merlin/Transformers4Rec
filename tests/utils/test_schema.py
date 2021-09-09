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

from transformers4rec.utils.schema import DatasetSchema
from transformers4rec.utils.tags import Tag


def test_schema_from_schema(schema_file):
    schema = DatasetSchema.from_proto(str(schema_file))

    assert len(schema.columns) == 18
    assert schema.columns[1].tags == ["list"]


def test_schema_from_yoochoose_schema(yoochoose_schema_file):
    schema = DatasetSchema.from_proto(str(yoochoose_schema_file))

    assert len(schema.columns) == 20
    assert len(schema.select_by_tag(Tag.CONTINUOUS).columns) == 6
    assert len(schema.select_by_tag(Tag.CATEGORICAL).columns) == 2


def test_schema_embedding_sizes_nvt(yoochoose_schema_file):
    pytest.importorskip("nvtabular")
    schema = DatasetSchema.from_proto(str(yoochoose_schema_file))

    assert schema.cardinalities() == {"item_id/list": 51996, "category/list": 332}
    embedding_sizes = schema.embedding_sizes_nvt(minimum_size=16, maximum_size=512)
    assert embedding_sizes == {"item_id/list": 512, "category/list": 41}


def test_schema_embedding_sizes(yoochoose_schema_file):
    schema = DatasetSchema.from_proto(str(yoochoose_schema_file))

    assert schema.cardinalities() == {"item_id/list": 51996, "category/list": 332}
    embedding_sizes = schema.embedding_sizes(multiplier=3.0)
    assert embedding_sizes == {"item_id/list": 46, "category/list": 13}
