import pytest
from merlin_sl.schema import schema


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
