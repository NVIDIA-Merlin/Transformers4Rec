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

from merlin_standard_lib.schema import schema


def test_column_proto_txt():
    col = schema.ColumnSchema(name="qa")

    assert col.to_proto_text() == 'name: "qa"\n'
    assert schema.Schema([col]).to_proto_text() == 'feature {\n  name: "qa"\n}\n'


def test_column_schema():
    col = schema.ColumnSchema(name="a")

    assert isinstance(col, schema.ColumnSchema) and col.name == "a"

    assert col.with_name("b").name == "b"
    assert set(col.with_tags(["tag_1", "tag_2"]).tags) == {"tag_1", "tag_2"}
    assert col.with_properties(dict(a=5)).properties == dict(a=5)


def test_schema():
    s = schema.Schema(
        [
            schema.ColumnSchema.create_continuous("con_1"),
            schema.ColumnSchema.create_continuous("con_2_int", is_float=False),
            schema.ColumnSchema.create_categorical("cat_1", 1000),
            schema.ColumnSchema.create_categorical(
                "cat_2", 100, value_count=schema.ValueCount(1, 20)
            ),
        ]
    )

    assert len(s.select_by_type(schema.FeatureType.INT).column_names) == 3
    assert len(s.select_by_name(lambda x: x.startswith("con")).column_names) == 2
    assert len(s.remove_by_name(lambda x: x.startswith("cat_1")).column_names) == 3

    new_tag = s.map_column_schemas(lambda col: col.with_tags(["new-tag"]))
    assert all("new-tag" in col.tags for col in new_tag.column_schemas)


def test_column_schema_categorical_with_shape():
    col = schema.ColumnSchema.create_categorical("cat_1", 1000, shape=(1,))

    assert col.shape.dim[0].size == 1

    assert schema.Schema([col]).categorical_cardinalities() == dict(cat_1=1000 + 1)


def test_column_schema_categorical_with_value_count():
    col = schema.ColumnSchema.create_categorical(
        "cat_1", 1000, value_count=schema.ValueCount(0, 100)
    )

    assert col.value_count.min == 0 and col.value_count.max == 100


@pytest.mark.parametrize("is_float", [True, False])
def test_column_schema_continuous_with_shape(is_float):
    col = schema.ColumnSchema.create_continuous(
        "con_1", min_value=0, max_value=10, shape=(1,), is_float=is_float
    )

    cast_fn = float if is_float else int
    domain = col.float_domain if is_float else col.int_domain
    assert domain.min == cast_fn(0) and domain.max == cast_fn(10)


@pytest.mark.parametrize("is_float", [True, False])
def test_column_schema_continuous_with_value_count(is_float):
    col = schema.ColumnSchema.create_continuous(
        "con_1", min_value=0, max_value=10, is_float=is_float, value_count=schema.ValueCount(0, 100)
    )

    assert col.value_count.min == 0 and col.value_count.max == 100

    cast_fn = float if is_float else int
    domain = col.float_domain if is_float else col.int_domain
    assert domain.min == cast_fn(0) and domain.max == cast_fn(10)
