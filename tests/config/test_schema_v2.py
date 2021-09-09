from transformers4rec.config import schema_v2 as schema


def test_column_schema():
    col = schema.ColumnSchema(name="a")

    assert isinstance(col, schema.ColumnSchema) and col.name == "a"

    assert col.with_name("b").name == "b"
    assert set(col.with_tags(["tag_1", "tag_2"]).tags) == {"tag_1", "tag_2"}


def test_column_schema_categorical_with_shape():
    col = schema.ColumnSchema.create_categorical("cat_1", 1000, shape=(1,))

    assert col.shape.dim[0].size == 1


def test_column_schema_categorical_with_value_count():
    col = schema.ColumnSchema.create_categorical(
        "cat_1", 1000, value_count=schema.ValueCount(0, 100)
    )

    assert col.value_count.min == 0 and col.value_count.max == 100
